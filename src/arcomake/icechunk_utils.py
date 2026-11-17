# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: MIT
"""Icechunk repositories: every output of arco-make is one, written in commits on branch `main`.

A path names a repository on a local filesystem (a directory, or a `file://` URL), on S3
(`s3://bucket/prefix`) or on Google Cloud Storage (`gs://bucket/prefix`). Recipes give its options
in `storage_options`, as for fsspec, but only these are understood (others are an error):
- S3: `profile` (else `AWS_PROFILE`), `region` (else the bucket's, asked to S3 at the endpoint;
  else the profile's), `endpoint_url`, `anon`; credentials come from botocore, as for every other
  S3 access of arco-make (icechunk's own AWS client cannot use some profiles), refreshed before
  they expire.
- GCS: `token = "anon"` for anonymous access, or the path of a service account file; otherwise
  the environment's credentials.
"""

import datetime
import functools
import logging
import pathlib
from collections.abc import Hashable, Mapping
from typing import Any, Literal

import fsspec.utils
import icechunk
import xarray as xr
from icechunk.xarray import to_icechunk

logger = logging.getLogger(__name__)

BRANCH = "main"
S3_OPTIONS = {"profile", "region", "endpoint_url", "anon"}
GCS_OPTIONS = {"token"}


def _botocore_credentials(profile: str | None) -> icechunk.S3StaticCredentials:
  """Credentials of `profile` (or the environment's), valid for at most 10 minutes: icechunk
  asks again when they expire, and botocore refreshes temporary credentials as needed."""
  import botocore.session

  credentials = botocore.session.Session(profile=profile).get_credentials()
  if credentials is None:
    raise ValueError(f"no AWS credentials for profile {profile or '(environment)'}")
  frozen = credentials.get_frozen_credentials()
  soon = datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=10)
  expiry = getattr(credentials, "_expiry_time", None)
  return icechunk.S3StaticCredentials(
    access_key_id=frozen.access_key,
    secret_access_key=frozen.secret_key,
    session_token=frozen.token,
    expires_after=min(expiry, soon) if expiry is not None else soon,
  )


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
    if anonymous:
      return icechunk.s3_storage(
        bucket=bucket, prefix=prefix, region=region, endpoint_url=endpoint_url, anonymous=True
      )
    return icechunk.s3_storage(
      bucket=bucket,
      prefix=prefix,
      region=region,
      endpoint_url=endpoint_url,
      get_credentials=functools.partial(_botocore_credentials, profile),
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


def write(
  dataset: xr.Dataset,
  path: str | pathlib.Path,
  *,
  storage_options: Mapping[str, Any] | None = None,
  message: str,
  mode: Literal["w", "w-", "a", "a-", "r+"] = "w",
  append_dim: Hashable | None = None,
  group: str | None = None,
) -> str:
  """Write `dataset` (its variables' encodings set the layout) in one commit; return its id.

  The repository is created if missing. Mode "w" replaces the content in a new commit: earlier
  versions stay in the repository's history.
  """
  session = writable_session(path, storage_options)
  to_icechunk(dataset, session, mode=mode, append_dim=append_dim, group=group)
  snapshot = session.commit(message)
  logger.info(f"Committed {snapshot} to {path}: {message}")
  return snapshot


def writable_session(
  path: str | pathlib.Path, storage_options: Mapping[str, Any] | None = None
) -> icechunk.Session:
  """A writable session on `main` of the repository at `path` (created if missing)."""
  return icechunk.Repository.open_or_create(storage(path, storage_options)).writable_session(BRANCH)


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
