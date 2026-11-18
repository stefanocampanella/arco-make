# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Icechunk repositories: every output of arco-make is one, written in commits on branch `main`.

A path names a repository on a local filesystem (a directory, or a `file://` URL), on S3
(`s3://bucket/prefix`) or on Google Cloud Storage (`gs://bucket/prefix`). Recipes give its options
in `storage_options`, as for fsspec, but only these are understood (others are an error):
- S3: `profile` (else `AWS_PROFILE`), `region` (else the bucket's, asked to S3 at the endpoint;
  else the profile's), `endpoint_url`, `anon`, `network_stream_timeout_seconds` (default 20, see
  `STREAM_TIMEOUT`); credentials come from botocore, as for every other
  S3 access of arco-make (icechunk's own AWS client cannot use some profiles), refreshed before
  they expire.
- GCS: `token = "anon"` for anonymous access, or the path of a service account file; otherwise
  the environment's credentials.
"""

import dataclasses
import datetime
import functools
import logging
import pathlib
import threading
from collections.abc import Hashable, Mapping
from typing import Any, Literal

import fsspec.utils
import icechunk
import xarray as xr
from icechunk.xarray import to_icechunk

from arcomake import sharding

logger = logging.getLogger(__name__)

BRANCH = "main"
# Commit metadata key of the run a commit belongs to (see `Run`).
RUN_KEY = "arcomake_run"
S3_OPTIONS = {"profile", "region", "endpoint_url", "anon", "network_stream_timeout_seconds"}
# Seconds without progress before an S3 request is retried (icechunk's default: 60). Requests
# stall now and then: on the project bucket, whole reads that take 6-7 s took up to 107 s with
# 60 s, at most 22 s with 10 s. The timeout is reset by progress, so it only cuts stalls.
STREAM_TIMEOUT = 20
GCS_OPTIONS = {"token"}


# Credentials resolved in this process, per profile: resolving them again for every client or
# fork icechunk creates (a botocore session each, possibly an external credential process) made
# writes to S3 several times slower.
_CREDENTIALS: dict[str | None, icechunk.S3StaticCredentials] = {}
_CREDENTIALS_LOCK = threading.Lock()


def _botocore_credentials(profile: str | None) -> icechunk.S3StaticCredentials:
  """Credentials of `profile` (or the environment's), valid for at most 10 minutes: icechunk
  asks again when they expire, and botocore refreshes temporary credentials as needed. Reused
  while more than a minute of validity is left."""
  now = datetime.datetime.now(datetime.UTC)
  with _CREDENTIALS_LOCK:
    cached = _CREDENTIALS.get(profile)
    if (
      cached is not None
      and cached.expires_after is not None
      and cached.expires_after - now > datetime.timedelta(minutes=1)
    ):
      return cached
    import botocore.session

    credentials = botocore.session.Session(profile=profile).get_credentials()
    if credentials is None:
      raise ValueError(f"no AWS credentials for profile {profile or '(environment)'}")
    frozen = credentials.get_frozen_credentials()
    soon = now + datetime.timedelta(minutes=10)
    expiry = getattr(credentials, "_expiry_time", None)
    _CREDENTIALS[profile] = icechunk.S3StaticCredentials(
      access_key_id=frozen.access_key,
      secret_access_key=frozen.secret_key,
      session_token=frozen.token,
      expires_after=min(expiry, soon) if expiry is not None else soon,
    )
    return _CREDENTIALS[profile]


@functools.cache
def _bucket_region(
  bucket: str, profile: str | None, endpoint_url: str | None, anonymous: bool
) -> str:
  """The region of an S3 bucket, as S3 reports it (`x-amz-bucket-region`, also in the error
  answers, e.g. 301 or 403); else the profile's; else, for an S3-compatible endpoint, us-east-1.

  Asked to the recipe's endpoint, signed or not as the repository's requests will be.
  """
  import botocore
  import botocore.config
  import botocore.exceptions
  import botocore.session

  session = botocore.session.Session(profile=profile)
  configured = session.get_config_variable("region")
  config = botocore.config.Config(signature_version=botocore.UNSIGNED) if anonymous else None
  client = session.create_client(
    "s3", endpoint_url=endpoint_url, region_name=configured or "us-east-1", config=config
  )
  try:
    headers = client.head_bucket(Bucket=bucket)["ResponseMetadata"]["HTTPHeaders"]
  except botocore.exceptions.ClientError as error:
    headers = error.response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
  region = headers.get("x-amz-bucket-region") or configured
  if region:
    return str(region)
  if endpoint_url is not None:
    return "us-east-1"
  raise ValueError(f"the region of bucket {bucket!r} is unknown: set storage_options.region")


def _unknown(options: Mapping[str, Any], known: set[str], where: str) -> None:
  unknown = sorted(set(options) - known)
  if unknown:
    raise ValueError(
      f"storage_options {unknown} are not understood for {where} (known: {sorted(known)})"
    )


def storage(
  path: str | pathlib.Path, storage_options: Mapping[str, Any] | None = None
) -> icechunk.Storage:
  """The Icechunk storage of a repository path (see the module's docstring).

  Raises:
    ValueError: for a path Icechunk cannot store a repository at, an option it cannot honour, or
      an S3 bucket whose region cannot be found.
  """
  options = dict(storage_options or {})
  path = str(path)
  protocol = fsspec.utils.get_protocol(path)
  if protocol in ("file", "local"):
    return icechunk.local_filesystem_storage(
      str(pathlib.Path(path.removeprefix("file://")).absolute())
    )
  if protocol not in ("s3", "s3a", "gs", "gcs"):
    raise ValueError(f"{path}: Icechunk repositories live on local paths, s3:// or gs:// URLs")
  bucket, _, prefix = path.split("://", 1)[1].partition("/")
  prefix = prefix.strip("/") or None
  if protocol in ("s3", "s3a"):
    _unknown(options, S3_OPTIONS, "S3 repositories")
    anonymous = bool(options.get("anon"))
    profile, endpoint_url = options.get("profile"), options.get("endpoint_url")
    region = options.get("region") or _bucket_region(bucket, profile, endpoint_url, anonymous)
    timeout = int(options.get("network_stream_timeout_seconds", STREAM_TIMEOUT))
    if anonymous:
      return icechunk.s3_storage(
        bucket=bucket,
        prefix=prefix,
        region=region,
        endpoint_url=endpoint_url,
        anonymous=True,
        network_stream_timeout_seconds=timeout,
      )
    return icechunk.s3_storage(
      bucket=bucket,
      prefix=prefix,
      region=region,
      endpoint_url=endpoint_url,
      network_stream_timeout_seconds=timeout,
      get_credentials=functools.partial(_botocore_credentials, profile),
      # The credentials travel with the storage to forks and workers, which ask again only when
      # they expire.
      scatter_initial_credentials=True,
    )
  _unknown(options, GCS_OPTIONS, "GCS repositories")
  token = options.get("token")
  if token == "anon":
    return icechunk.gcs_storage(bucket=bucket, prefix=prefix, anonymous=True)
  if token is None:
    return icechunk.gcs_storage(bucket=bucket, prefix=prefix, from_env=True)
  if isinstance(token, str) and pathlib.Path(token).expanduser().is_file():
    return icechunk.gcs_storage(
      bucket=bucket, prefix=prefix, service_account_file=str(pathlib.Path(token).expanduser())
    )
  raise ValueError(
    'storage_options.token for GCS repositories: "anon" or a service account file '
    "(without it, the environment's credentials)"
  )


def is_repository(
  path: str | pathlib.Path, storage_options: Mapping[str, Any] | None = None
) -> bool:
  """Whether `path` holds an Icechunk repository."""
  return icechunk.Repository.exists(storage(path, storage_options))


def has_no_version(repository_storage: icechunk.Storage) -> bool:
  """Whether `main` of the repository is still its first snapshot (no run finished there)."""
  tip = next(iter(icechunk.Repository.open(repository_storage).ancestry(branch=BRANCH)))
  return tip.parent_id is None


def run_label(command: str) -> str:
  """A label for one run of `command`: its start time, to the microsecond (branch names, commit
  metadata, and the messages that tie the outputs of one run together)."""
  return f"{command}-{datetime.datetime.now(datetime.UTC):%Y%m%dT%H%M%S%fZ}"


def write(
  dataset: xr.Dataset,
  path: str | pathlib.Path,
  *,
  storage_options: Mapping[str, Any] | None = None,
  message: str,
  mode: Literal["w", "w-", "a", "a-", "r+"] = "w",
  append_dim: Hashable | None = None,
  group: str | None = None,
  retention: "Retention | None" = None,
  command: str = "write",
  align_chunks: bool = False,
  shard_order: sharding.ShardOrder = sharding.DEFAULT,
) -> str:
  """Write `dataset` in a run of `command` of a single commit (see `Run`); return the new tip of
  `main`.

  The repository is created if missing. Mode "w" replaces the content in the new version: earlier
  versions stay in the repository's history until `retention` expires them.
  """
  run = Run(path, storage_options, run_label(command))
  run.write(
    dataset,
    message=message,
    mode=mode,
    append_dim=append_dim,
    group=group,
    align_chunks=align_chunks,
    shard_order=shard_order,
  )
  return run.finish(retention)


def readonly_store(
  path: str | pathlib.Path,
  storage_options: Mapping[str, Any] | None = None,
  *,
  branch: str | None = None,
  tag: str | None = None,
  snapshot_id: str | None = None,
) -> icechunk.IcechunkStore:
  """The Zarr store of a version of the repository at `path` (by default, the tip of `main`)."""
  repo = icechunk.Repository.open(storage(path, storage_options))
  if tag is None and snapshot_id is None:
    branch = branch or BRANCH
  return repo.readonly_session(branch=branch, tag=tag, snapshot_id=snapshot_id).store


def fsspec_options(
  path: str | pathlib.Path, storage_options: Mapping[str, Any] | None
) -> dict[str, Any]:
  """The fsspec equivalent of a repository's `storage_options` (to check whether `path` exists)."""
  options = dict(storage_options or {})
  if fsspec.utils.get_protocol(str(path)) in ("s3", "s3a") and "region" in options:
    options.setdefault("client_kwargs", {})["region_name"] = options.pop("region")
  return options


# Commit metadata of the blocks of a download (see `WaitingRun`): the block's interval, and a
# fingerprint of the recipe it was written with.
BLOCK_KEY = "arcomake_block"
RECIPE_KEY = "arcomake_recipe"


@dataclasses.dataclass(frozen=True)
class WaitingRun:
  """A run that failed before `main` moved to it: its branch holds the blocks it committed."""

  label: str
  branch: str
  base: str  # the tip of `main` when it started
  blocks: tuple[str, ...]  # the intervals committed, oldest first (`BLOCK_KEY`)
  recipe: str | None  # the fingerprint of the recipe its blocks were written with
  current: bool  # whether `main` is still where the run started (it can still be promoted)


def waiting_run(
  path: str | pathlib.Path, storage_options: Mapping[str, Any] | None, command: str
) -> WaitingRun | None:
  """The newest run of `command` whose branch is still in the repository at `path` (a run that
  failed), or `None`; also `None` if there is no repository."""
  repository_storage = storage(path, storage_options)
  if not icechunk.Repository.exists(repository_storage):
    return None
  repo = icechunk.Repository.open(repository_storage)
  branches = sorted(b for b in repo.list_branches() if b.startswith(f"run/{command}-"))
  if not branches:
    return None
  branch = branches[-1]
  label = branch.removeprefix("run/")
  blocks: list[str] = []
  recipe = None
  base = None
  for snapshot in repo.ancestry(branch=branch):
    metadata = snapshot.metadata or {}
    if metadata.get(RUN_KEY) != label:
      base = snapshot.id
      break
    if metadata.get(BLOCK_KEY) is not None:
      blocks.append(metadata[BLOCK_KEY])
    recipe = recipe or metadata.get(RECIPE_KEY)
  assert base is not None  # a run's branch starts from a snapshot of `main`
  current = repo.lookup_branch(BRANCH) == base
  return WaitingRun(label, branch, base, tuple(reversed(blocks)), recipe, current)


class PromotionError(RuntimeError):
  """`main` changed during a run: its output was not promoted (it stays on the run's branch)."""


class Run:
  """The writes of one run of a command to one repository, on a branch of their own.

  The branch (`run/<label>`) starts at the tip of `main`; every write of the run is a commit on
  it, with the run's label in the commit metadata. `finish` moves `main` to the run's last commit
  if `main` has not changed meanwhile, deletes the branch, then applies the recipe's retention.
  A run that fails before `finish` leaves `main` as it was (its branch stays, with what was
  written).
  """

  def __init__(
    self,
    path: str | pathlib.Path,
    storage_options: Mapping[str, Any] | None,
    label: str,
    *,
    waiting: "WaitingRun | None" = None,
  ) -> None:
    """Start a run labelled `label`, or continue `waiting` (a run that failed, see
    `waiting_run`) on its branch."""
    self.path = str(path)
    self.repo = icechunk.Repository.open_or_create(storage(path, storage_options))
    if waiting is not None:
      self.label, self.branch, self.base = waiting.label, waiting.branch, waiting.base
      logger.info(f"Continuing {self.path} on branch {self.branch} (from {self.base})")
      return
    self.label = label
    self.branch = f"run/{label}"
    self.base = self.repo.lookup_branch(BRANCH)
    self.repo.create_branch(self.branch, self.base)
    logger.info(f"Writing {self.path} on branch {self.branch} (from {self.base})")

  def session(self) -> icechunk.Session:
    return self.repo.writable_session(self.branch)

  def commit(
    self, session: icechunk.Session, message: str, metadata: Mapping[str, Any] | None = None
  ) -> str:
    """Commit `session` on the run's branch; the run's label (and `metadata`) go in the commit's
    metadata."""
    snapshot = str(session.commit(message, metadata={**(metadata or {}), RUN_KEY: self.label}))
    logger.info(f"Committed {snapshot} to {self.path} ({self.branch}): {message}")
    return snapshot

  def write(
    self,
    dataset: xr.Dataset,
    *,
    message: str,
    mode: Literal["w", "w-", "a", "a-", "r+"] = "w",
    append_dim: Hashable | None = None,
    group: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    align_chunks: bool = False,
    shard_order: sharding.ShardOrder = sharding.DEFAULT,
  ) -> str:
    """Write `dataset` (its variables' encodings set the layout) in one commit of the run. With
    `align_chunks`, Dask chunks that straddle stored chunks are rechunked to fit them (xarray's,
    region-aware: an append may start inside a chunk); else such chunks fail the write.
    `shard_order`: the order of the chunks inside shards (see `sharding`)."""
    session = self.session()
    with sharding.write_order(shard_order):
      if sharding.creates_arrays(shard_order, mode):
        # Metadata first: values written into arrays zarr opens follow the order.
        dataset.to_zarr(
          session.store, mode=mode, group=group, compute=False, consolidated=False, zarr_format=3
        )
        mode = "r+"
      to_icechunk(
        dataset, session, mode=mode, append_dim=append_dim, group=group, align_chunks=align_chunks
      )
    return self.commit(session, message, metadata)

  def last_block(self) -> str | None:
    """The block (`BLOCK_KEY`) the tip of the run's branch commits, if it is one of this run's:
    a commit whose answer was lost (a timeout after it landed) is not redone."""
    tip = next(iter(self.repo.ancestry(branch=self.branch)))
    metadata = tip.metadata or {}
    block = metadata.get(BLOCK_KEY)
    return str(block) if block is not None and metadata.get(RUN_KEY) == self.label else None

  def finish(self, retention: "Retention | None" = None) -> str:
    """Move `main` to the run's last commit (if `main` is still where the run started), then
    delete the run's branch and apply `retention`; return the new tip of `main`.

    Once `main` has moved the run has succeeded: errors deleting the branch or applying the
    retention are logged, not raised.

    Raises:
      PromotionError: if `main` changed during the run; the run's branch is kept.
    """
    tip = self.repo.lookup_branch(self.branch)
    try:
      self.repo.reset_branch(BRANCH, tip, from_snapshot_id=self.base)
    except icechunk.ConflictError as error:
      raise PromotionError(
        f"{self.path}: main changed during the run, which stays on branch {self.branch}"
      ) from error
    logger.info(f"{self.path}: main is now {tip} (run {self.label})")
    try:
      self.repo.delete_branch(self.branch)
    except Exception:
      logger.exception(f"{self.path}: branch {self.branch} could not be deleted (main is promoted)")
    if retention is not None:
      retention.apply(self.repo, self.path, current=self.label)
    return tip


# Objects (chunks, manifests) younger than this are never collected, nor are the branches of
# other runs whose tips are younger deleted: a run still writing to the same repository has
# uncommitted chunks no snapshot references yet, and a branch it is still committing to.
GRACE = datetime.timedelta(days=7)


class Retention:
  """How long the versions of `main` replaced by later runs are kept (`[save] keep_history` or
  `keep_versions`), applied after each successful run.

  `keep_history`: a version is kept while it was replaced less than that long ago (when the run
  after it was written). `keep_versions`: the number of versions kept, the current one included.
  Older snapshots are expired; then, if anything was, the objects no snapshot references and older
  than `GRACE` are deleted, with the branches of other runs whose tips are older than both the
  limit and `GRACE` (failed runs). The current version is never expired.
  """

  def __init__(
    self, keep_history: datetime.timedelta | None = None, keep_versions: int | None = None
  ) -> None:
    if (keep_history is None) == (keep_versions is None):
      raise ValueError("give exactly one of keep_history and keep_versions")
    self.keep_history = keep_history
    self.keep_versions = keep_versions

  def limit(self, repo: icechunk.Repository) -> datetime.datetime | None:
    """Snapshots written before this time are expired; `None`: nothing to expire."""
    # The versions of `main`, newest first: the last commit of each run in its history. Only as
    # many as needed are read (a download has a commit per block).
    tips: list[Any] = []
    seen: set[str] = set()
    now = datetime.datetime.now(datetime.UTC)
    for snapshot in repo.ancestry(branch=BRANCH):
      run = (snapshot.metadata or {}).get(RUN_KEY)
      if run is None or run in seen:
        continue
      seen.add(run)
      tips.append(snapshot)
      if self.keep_versions is not None and len(tips) > self.keep_versions:
        break
      # tips[-1] was replaced when tips[-2] was written: past the limit, so are the older ones.
      if (
        self.keep_history is not None
        and len(tips) > 1
        and now - tips[-2].written_at > self.keep_history
      ):
        break
    if self.keep_versions is not None:
      kept = tips[: self.keep_versions]
    else:
      kept = tips[:1] + [
        older
        for newer, older in zip(tips, tips[1:], strict=False)
        if now - newer.written_at <= self.keep_history  # type: ignore[operator]
      ]
    if len(kept) == len(tips):
      return None
    oldest_kept = kept[-1].written_at
    assert isinstance(oldest_kept, datetime.datetime)
    return oldest_kept

  def apply(self, repo: icechunk.Repository, path: str, *, current: str | None = None) -> None:
    """Expire and collect; errors are logged, not raised (the run is already promoted)."""
    try:
      limit = self.limit(repo)
      if limit is None:
        return
      safe = min(limit, datetime.datetime.now(datetime.UTC) - GRACE)
      for branch in repo.list_branches():
        if branch == BRANCH or branch == f"run/{current}":
          continue
        tip = next(iter(repo.ancestry(branch=branch)))
        if tip.written_at < safe:
          repo.delete_branch(branch)
          logger.info(f"{path}: deleted branch {branch} (a run without commits since {safe:%F})")
      expired = repo.expire_snapshots(limit)
      if not expired:
        return
      summary = repo.garbage_collect(safe)
      logger.info(
        f"{path}: expired {len(expired)} snapshots older than {limit:%Y-%m-%dT%H:%M:%SZ};"
        f" deleted {summary.chunks_deleted} chunks, {summary.manifests_deleted} manifests,"
        f" {summary.snapshots_deleted} snapshots older than {safe:%Y-%m-%dT%H:%M:%SZ}"
      )
    except Exception:
      logger.exception(f"{path}: the retention could not be applied; the output is complete")
