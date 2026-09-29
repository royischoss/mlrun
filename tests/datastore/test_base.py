# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import string
import tempfile
from collections import Counter
from contextlib import nullcontext as does_not_raise
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import fsspec
import pandas as pd
import pyarrow
import pytest
import pytz
import storey.utils
from fsspec.implementations.memory import MemoryFileSystem

import mlrun.datastore
import mlrun.datastore.targets
import mlrun.feature_store
from mlrun import MLRunInvalidArgumentError, new_function
from mlrun.datastore import KafkaSource
from mlrun.datastore.azure_blob import AzureBlobStore
from mlrun.datastore.base import HttpStore
from mlrun.datastore.datastore import schema_to_model_provider, schema_to_store
from mlrun.datastore.datastore_profile import DatastoreProfileKafkaStream
from mlrun.datastore.dbfs_store import DBFSStore
from mlrun.datastore.filestore import FileStore
from mlrun.datastore.google_cloud_storage import GoogleCloudStorageStore
from mlrun.datastore.model_provider.huggingface_provider import HuggingFaceProvider
from mlrun.datastore.model_provider.openai_provider import OpenAIProvider
from mlrun.datastore.redis import RedisStore
from mlrun.datastore.s3 import S3Store
from mlrun.datastore.v3io import V3ioStore


def test_http_fs_parquet_as_df():
    data_item = mlrun.datastore.store_manager.object(
        "https://s3.wasabisys.com/iguazio/data/market-palce/aggregate/metrics.pq"
    )
    data_item.as_df()


def test_http_fs_parquet_with_params_as_df():
    data_item = mlrun.datastore.store_manager.object(
        "https://s3.wasabisys.com/iguazio/data/market-palce/aggregate/metrics.pq?param1=1&param2=2"
    )
    data_item.as_df()


def test_s3_fs_parquet_as_df():
    data_item = mlrun.datastore.store_manager.object(
        "s3://coiled-datasets/timeseries/20-years/parquet/part.0.parquet"
    )
    data_item.as_df()


def test_kafka_source_with_attributes():
    source = KafkaSource(
        brokers="broker_host:9092",
        topics="mytopic",
        group="mygroup",
        sasl_user="myuser",
        sasl_pass="mypassword",
        attributes={
            "sasl": {
                "handshake": True,
            },
        },
    )
    function = new_function(kind="remote")
    source.add_nuclio_trigger(function)
    attributes = function.spec.config["spec.triggers.kafka"]["attributes"]
    assert attributes["brokers"] == ["broker_host:9092"]
    assert attributes["topics"] == ["mytopic"]
    assert attributes["consumerGroup"] == "mygroup"
    assert attributes["sasl"] == {
        "enable": True,
        "user": "myuser",
        "password": "mypassword",
        "mechanism": "PLAIN",
        "handshake": True,
    }


def test_kafka_source_with_attributes_as_ds_profile():
    ds = DatastoreProfileKafkaStream(
        name="dskafkasrc",
        brokers="broker_host:9092",
        topics="mytopic",
        group="mygroup",
        sasl_user="myuser",
        sasl_pass="mypassword",
        kwargs_public={
            "sasl": {
                "handshake": True,
            },
        },
        kwargs_private={
            "sasl": {
                "password": "wrong_password",
            },
        },
    )
    source = KafkaSource(path="ds://dskafkasrc")
    function = new_function(kind="remote")
    mlrun.datastore.sources.datastore_profile_read = Mock(return_value=ds)
    source.add_nuclio_trigger(function)
    attributes = function.spec.config["spec.triggers.kafka"]["attributes"]
    assert attributes["brokers"] == ["broker_host:9092"]
    assert attributes["topics"] == ["mytopic"]
    assert attributes["consumerGroup"] == "mygroup"
    assert attributes["sasl"] == {
        "enable": True,
        "user": "myuser",
        "password": "mypassword",
        "mechanism": "PLAIN",
        "handshake": True,
    }


def test_kafka_source_with_attributes_as_ds_profile_brokers_list():
    ds = DatastoreProfileKafkaStream(
        name="dskafkasrc",
        brokers=["broker_host:9092", "broker_host2:9093"],
        topics=["mytopic", "mytopic2"],
        group="mygroup",
        kwargs_public={
            "sasl": {
                "handshake": True,
                "enabled": True,
            },
        },
        kwargs_private={
            "sasl": {
                "password": "mypassword",
                "user": "myuser",
            },
        },
    )
    source = KafkaSource(path="ds://dskafkasrc")
    function = new_function(kind="remote")
    mlrun.datastore.sources.datastore_profile_read = Mock(return_value=ds)
    source.add_nuclio_trigger(function)
    attributes = function.spec.config["spec.triggers.kafka"]["attributes"]
    assert attributes["brokers"] == ["broker_host:9092", "broker_host2:9093"]
    assert attributes["topics"] == ["mytopic", "mytopic2"]
    assert attributes["consumerGroup"] == "mygroup"
    assert attributes["sasl"] == {
        "enabled": True,
        "user": "myuser",
        "password": "mypassword",
        "handshake": True,
    }


def test_kafka_source_without_attributes():
    source = KafkaSource(
        brokers="broker_host:9092",
        topics="mytopic",
        group="mygroup",
        sasl_user="myuser",
        sasl_pass="mypassword",
    )
    function = new_function(kind="remote")
    source.add_nuclio_trigger(function)
    attributes = function.spec.config["spec.triggers.kafka"]["attributes"]
    assert attributes["brokers"] == ["broker_host:9092"]
    assert attributes["topics"] == ["mytopic"]
    assert attributes["consumerGroup"] == "mygroup"
    assert attributes["sasl"] == {
        "enable": True,
        "handshake": True,
        "user": "myuser",
        "password": "mypassword",
        "mechanism": "PLAIN",
    }


@pytest.mark.parametrize(
    "schemas,expected_class,expected",
    [
        (["file"] + list(string.ascii_lowercase), FileStore, does_not_raise()),
        (["s3"], S3Store, does_not_raise()),
        (["az", "wasbs", "wasb"], AzureBlobStore, does_not_raise()),
        (["v3io", "v3ios"], V3ioStore, does_not_raise()),
        (["redis", "rediss"], RedisStore, does_not_raise()),
        (["http", "https"], HttpStore, does_not_raise()),
        (["gcs", "gs"], GoogleCloudStorageStore, does_not_raise()),
        (["dbfs"], DBFSStore, does_not_raise()),
        (["random"], None, pytest.raises(ValueError)),
    ],
)
def test_schema_to_store(schemas, expected_class, expected):
    with expected:
        stores = [schema_to_store(schema) for schema in schemas]
        assert all(store == expected_class for store in stores)


@pytest.mark.parametrize(
    "schemas,expected_class,expected",
    [
        (["openai"], OpenAIProvider, does_not_raise()),
        (["huggingface"], HuggingFaceProvider, does_not_raise()),
        (["random"], None, pytest.raises(ValueError)),
    ],
)
def test_schema_to_model_provider(schemas, expected_class, expected):
    with expected:
        stores = [schema_to_model_provider(schema) for schema in schemas]
        assert all(store == expected_class for store in stores)


# ML-6308
@pytest.mark.parametrize("start_time_tz", [None, "naive", "with_tz"])
@pytest.mark.parametrize("end_time_tz", [None, "naive", "with_tz"])
@pytest.mark.parametrize("df_tz", [False, True])
def test_as_df_time_filters(start_time_tz, end_time_tz, df_tz):
    time_column = "timestamp"

    parquet_file = os.path.join(
        os.path.join(os.path.dirname(__file__), "assets"), "testdata.parquet"
    )
    full_df = pd.read_parquet(parquet_file)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".parquet", delete=True
    ) as parquet_file:
        if df_tz:
            full_df[time_column] = full_df[time_column].dt.tz_localize("UTC")
        full_df.to_parquet(parquet_file.name)

        data_item = mlrun.datastore.store_manager.object(f"file://{parquet_file.name}")

        start_time = None
        tzinfo = pytz.UTC if start_time_tz == "with_tz" else None
        if start_time_tz:
            start_time = datetime(2020, 12, 1, 17, 28, 15, tzinfo=tzinfo)

        end_time = None
        tzinfo = pytz.UTC if end_time_tz == "with_tz" else None
        if end_time_tz:
            end_time = datetime(2020, 12, 1, 17, 29, 15, tzinfo=tzinfo)

        if {start_time_tz, end_time_tz} == {"naive", "with_tz"}:
            expectation = pytest.raises(
                MLRunInvalidArgumentError,
                match="start_time and end_time must have the same time zone",
            )
        else:
            expectation = does_not_raise()

        with expectation:
            resp = data_item.as_df(
                start_time=start_time, end_time=end_time, time_column=time_column
            )
            num_row_expected = (
                190 - (80 if start_time_tz else 0) - (90 if end_time_tz else 0)
            )
            assert len(resp) == num_row_expected


def test_partition_filtering_year_month():
    """
    Simple test for partition filtering with year/month structure.
    Tests filtering data from 2020-2022 with year=YYYY/month=MM partitions.
    """
    time_column = "timestamp"

    # Create test data spanning 2020-2022, all months
    test_data = []
    for year in range(2020, 2023):  # 2020, 2021, 2022
        for month in range(1, 13):  # Jan-Dec
            for day in [5, 15, 25]:  # 3 records per month
                timestamp = datetime(year, month, day, 12, 0, 0)
                test_data.append(
                    {
                        "timestamp": timestamp,
                        "year": year,
                        "month": month,
                        "value": year * 100 + month * 10 + day,
                    }
                )

    df = pd.DataFrame(test_data)

    # Create partitioned parquet structure: year=YYYY/month=MM/
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write partitioned parquet files
        df.to_parquet(temp_dir, partition_cols=["year", "month"], engine="pyarrow")

        # Create DataItem with trailing slash for directory
        dir_url = f"file://{temp_dir}/"
        data_item = mlrun.datastore.store_manager.object(dir_url)

        # Test filter: Get data from June 2021 to August 2021
        start_time = datetime(2021, 6, 10, 0, 0, 0)  # June 10, 2021
        end_time = datetime(2021, 8, 20, 0, 0, 0)  # August 20, 2021

        # Execute the filter
        result = data_item.as_df(
            start_time=start_time,
            end_time=end_time,
            time_column=time_column,
            format="parquet",
            optimize_discovery=False,
        )

        # Calculate expected results manually
        # Should include: June 15 & 25 + July 5, 15 & 25 + August 5 & 15 = 7 records
        expected_df = df[
            (df[time_column] > start_time) & (df[time_column] <= end_time)
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            result,
            expected_df,
            check_like=True,
            check_dtype=False,
            check_categorical=False,
        )


@pytest.mark.parametrize(
    "with_time_zone",
    [True, False],
)
@pytest.mark.parametrize(
    "description, start_time_args, end_time_args, partition_args, expected",
    [
        # ----------------------------
        # YEAR ONLY PARTITIONS
        # ----------------------------
        (
            "Year fully inside range",
            (2024, 1, 1, 0, 0),
            (2024, 12, 31, 23, 59),
            dict(year=2024),
            True,
        ),
        (
            "Year fully before range",
            (2024, 1, 1, 0, 0),
            (2024, 12, 31, 23, 59),
            dict(year=2023),
            False,
        ),
        (
            "Year fully after range",
            (2024, 1, 1, 0, 0),
            (2024, 12, 31, 23, 59),
            dict(year=2025),
            False,
        ),
        (
            "Year boundary overlap",
            (2023, 12, 31, 23, 59),
            (2024, 1, 1, 0, 0),
            dict(year=2024),
            True,
        ),
        # ----------------------------
        # YEAR + MONTH PARTITIONS
        # ----------------------------
        (
            "Month inside full-year range",
            (2024, 1, 1, 0, 0),
            (2024, 12, 31, 23, 59),
            dict(year=2024, month=2),
            True,
        ),
        (
            "Month outside (previous year)",
            (2024, 1, 1, 0, 0),
            (2024, 12, 31, 23, 59),
            dict(year=2023, month=12),
            False,
        ),
        (
            "Month crossing year boundary (Dec → Jan)",
            (2023, 12, 1, 0, 0),
            (2024, 1, 1, 0, 0),
            dict(year=2023, month=12),
            True,
        ),
        (
            "Exact boundary: range start == partition_end",
            (2024, 1, 1, 0, 0),
            (2024, 1, 1, 0, 0),
            dict(year=2024, month=1),
            True,
        ),
        # ----------------------------
        # YEAR + MONTH + DAY PARTITIONS
        # ----------------------------
        (
            "Day inside range",
            (2024, 5, 1, 0, 0),
            (2024, 12, 31, 0, 0),
            dict(year=2024, month=5, day=15),
            True,
        ),
        (
            "Day outside range",
            (2024, 5, 16, 0, 0),
            (2024, 12, 31, 0, 0),
            dict(year=2023, month=5, day=15),
            False,
        ),
        (
            "Day boundary exact-touch",
            (2024, 5, 15, 0, 0),
            (2024, 5, 16, 0, 0),
            dict(year=2024, month=5, day=15),
            True,
        ),
        # ----------------------------
        # YEAR + MONTH + DAY + HOUR PARTITIONS
        # ----------------------------
        (
            "Hour inside range",
            (2024, 6, 10, 4, 0),
            (2024, 6, 30, 0, 0),
            dict(year=2024, month=6, day=10, hour=5),
            True,
        ),
        (
            "Hour outside range",
            (2024, 6, 10, 6, 0),
            (2024, 6, 30, 0, 0),
            dict(year=2024, month=7, day=10, hour=5),
            False,
        ),
        (
            "Hour partition touches range start",
            (2024, 6, 10, 6, 0),
            (2024, 6, 10, 8, 0),
            dict(year=2024, month=6, day=10, hour=6),
            True,
        ),
        (
            "No start time",
            None,
            (2024, 6, 10, 8, 0),
            dict(year=2024, month=6, day=10, hour=6),
            True,
        ),
        (
            "No end time",
            (2024, 6, 10, 6, 0),
            None,
            dict(year=2024, month=6, day=10, hour=6),
            True,
        ),
    ],
)
def test_is_directory_in_range(
    description,
    start_time_args,
    end_time_args,
    partition_args,
    expected,
    with_time_zone,
):
    tz = pytz.UTC if with_time_zone else None

    start_time = datetime(*start_time_args, tzinfo=tz) if start_time_args else None
    end_time = datetime(*end_time_args, tzinfo=tz) if end_time_args else None

    result = mlrun.datastore.base.DataStore._is_directory_in_range(
        start_time, end_time, **partition_args
    )
    assert result == expected, f"Failed case: {description} (tz-aware={with_time_zone})"


class _CountingFileSystem(MemoryFileSystem):
    """In-memory filesystem that tallies the listing calls a read makes.

    Subclasses rather than wraps so it survives the isinstance checks pandas and pyarrow
    apply, and is cachable under its own protocol so the url-driven read inside
    `_read_partitioned_parquet` resolves back to this same instance.
    """

    protocol = "counting"
    cachable = True

    @classmethod
    def _strip_protocol(cls, path):
        if path.startswith("counting://"):
            path = "memory://" + path[len("counting://") :]
        return MemoryFileSystem._strip_protocol(path)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counts = Counter()

    def ls(self, path, detail=True, **kwargs):
        self.counts["ls"] += 1
        return super().ls(path, detail=detail, **kwargs)

    def info(self, path, **kwargs):
        self.counts["info"] += 1
        return super().info(path, **kwargs)

    def reset(self):
        self.counts = Counter()

    @property
    def listing_calls(self):
        return self.counts["ls"] + self.counts["info"]


fsspec.register_implementation("counting", _CountingFileSystem, clobber=True)

_COUNTING_UID = "ffd8dfe0a3b511ee"
_COUNTING_BASE = f"counting://bucket/parquet/key={_COUNTING_UID}"
# storey's window is (start, end], so rows are written half past the hour.
_COUNTING_TREE_END = datetime(2026, 9, 28, 11, 30)
_COUNTING_START = datetime(2026, 9, 28, 11, 0)
_COUNTING_END = datetime(2026, 9, 28, 11, 59, 59)
_HOUR_LAYOUT = ["year", "month", "day", "hour"]


@pytest.fixture
def counting_fs():
    _CountingFileSystem.clear_instance_cache()
    fs = fsspec.filesystem("counting")
    fs.store.clear()
    fs.pseudo_dirs[:] = [""]
    yield fs
    fs.store.clear()
    _CountingFileSystem.clear_instance_cache()


def _write_counting_tree(fs, hours: int) -> None:
    """One parquet file per hour, laid out as the model-monitoring offline target is."""
    for offset in range(hours):
        timestamp = _COUNTING_TREE_END - timedelta(hours=offset)
        path = (
            f"/bucket/parquet/key={_COUNTING_UID}"
            f"/year={timestamp.year}/month={timestamp.month:02d}"
            f"/day={timestamp.day:02d}/hour={timestamp.hour:02d}/data.parquet"
        )
        chunk = pd.DataFrame({"timestamp": [timestamp], "feature": [offset]})
        with fs.open(path, "wb") as handle:
            chunk.to_parquet(handle, index=False)


def _write_flat_counting_tree(fs, hours: int) -> None:
    """One parquet file per hour straight under the base, as an unpartitioned write leaves it."""
    for offset in range(hours):
        timestamp = _COUNTING_TREE_END - timedelta(hours=offset)
        path = f"/bucket/parquet/key={_COUNTING_UID}/chunk-{offset}.parquet"
        chunk = pd.DataFrame({"timestamp": [timestamp], "feature": [offset]})
        with fs.open(path, "wb") as handle:
            chunk.to_parquet(handle, index=False)


def _read_counting_tree(fs, partitions=None, url=_COUNTING_BASE):
    reader = mlrun.datastore.base.DataStore._parquet_reader(
        pd,
        url,
        fs,
        "timestamp",
        _COUNTING_START,
        _COUNTING_END,
        None,
        True,
        partitions,
    )
    fs.reset()
    return reader(url)


def test_supplied_partitions_skip_discovery(counting_fs, monkeypatch):
    """Supplying the layout returns the same rows without walking the store to find it."""
    _write_counting_tree(counting_fs, hours=24)

    discovered = _read_counting_tree(counting_fs)
    discovered_calls = counting_fs.listing_calls

    def _fail(*args, **kwargs):
        raise AssertionError(
            "find_partitions must not be called when partitions are given"
        )

    monkeypatch.setattr(storey.utils, "find_partitions", _fail)

    supplied = _read_counting_tree(
        counting_fs, partitions=["year", "month", "day", "hour"]
    )
    supplied_calls = counting_fs.listing_calls

    pd.testing.assert_frame_equal(supplied, discovered)
    assert len(supplied) == 1
    assert supplied_calls < discovered_calls


def test_supplied_partitions_cost_does_not_grow_with_the_tree(counting_fs):
    """Discovery cost stays flat as an endpoint ages, which is the property to protect."""
    partitions = ["year", "month", "day", "hour"]

    _write_counting_tree(counting_fs, hours=24)
    _read_counting_tree(counting_fs, partitions=partitions)
    one_day_calls = counting_fs.listing_calls

    _write_counting_tree(counting_fs, hours=720)
    result = _read_counting_tree(counting_fs, partitions=partitions)
    thirty_day_calls = counting_fs.listing_calls

    assert len(result) == 1
    assert thirty_day_calls == one_day_calls


def test_partitions_are_discovered_when_not_supplied(counting_fs, monkeypatch):
    """The fallback is untouched: without the input the store is still walked."""
    _write_counting_tree(counting_fs, hours=24)
    calls = []
    original = storey.utils.find_partitions

    def _spy(url, fs):
        calls.append(url)
        return original(url, fs)

    monkeypatch.setattr(storey.utils, "find_partitions", _spy)

    result = _read_counting_tree(counting_fs)

    assert calls == [_COUNTING_BASE]
    assert len(result) == 1


def test_supplied_partitions_ignored_for_a_single_file(counting_fs):
    """A single file has no levels below it, so a declared layout must not address any."""
    _write_counting_tree(counting_fs, hours=24)
    file_url = f"{_COUNTING_BASE}/year=2026/month=09/day=28/hour=11/data.parquet"

    result = _read_counting_tree(
        counting_fs, partitions=["year", "month", "day", "hour"], url=file_url
    )

    pd.testing.assert_frame_equal(
        result, _read_counting_tree(counting_fs, url=file_url)
    )
    assert len(result) == 1


def test_supplied_partitions_on_an_empty_directory(counting_fs):
    """An endpoint with nothing written yet yields an empty frame instead of an arrow error.

    Discovery finds no layout, falls back to a plain read and lets pyarrow fail on the
    missing time column. With the layout supplied the partitioned read matches nothing and
    returns empty, which is what MLRunEmptySampleDFError is there to represent.
    """
    counting_fs.makedirs(f"/bucket/parquet/key={_COUNTING_UID}", exist_ok=True)

    supplied = _read_counting_tree(
        counting_fs, partitions=["year", "month", "day", "hour"]
    )
    assert supplied.empty

    with pytest.raises(pyarrow.lib.ArrowInvalid):
        _read_counting_tree(counting_fs)


@pytest.mark.parametrize(
    "description, target_kwargs, expected",
    [
        (
            "declared granularity is forwarded",
            dict(time_partitioning_granularity="hour"),
            ["year", "month", "day", "hour"],
        ),
        (
            "a coarser granularity yields fewer levels",
            dict(time_partitioning_granularity="day"),
            ["year", "month", "day"],
        ),
        (
            "partitioned defaults to True for a directory and must not be guessed from",
            dict(),
            None,
        ),
        (
            "extra partition columns add levels that do not map onto the time units",
            dict(time_partitioning_granularity="hour", partition_cols=["region"]),
            None,
        ),
        (
            "key bucketing adds a level above the time units",
            dict(time_partitioning_granularity="hour", key_bucketing_number=0),
            None,
        ),
        (
            "the storey writer drops the time units when partitioning is off",
            dict(partitioned=False, time_partitioning_granularity="hour"),
            None,
        ),
    ],
)
def test_parquet_target_forwards_only_a_declared_layout(
    description, target_kwargs, expected
):
    target = mlrun.datastore.targets.ParquetTarget(
        path="counting://bucket/parquet", **target_kwargs
    )
    data_item = Mock()
    data_item.as_df.return_value = pd.DataFrame({"feature": [1]})

    with patch("mlrun.get_dataitem", return_value=data_item):
        target.as_df(start_time=_COUNTING_START, end_time=_COUNTING_END)

    assert data_item.as_df.call_args.kwargs.get("partitions") == expected, description


def test_parquet_target_read_skips_discovery_end_to_end(monkeypatch):
    """The layout reaches the reader through the whole target chain, not just in isolation.

    Covers ParquetTarget.as_df -> get_dataitem -> DataStore.as_df -> _parquet_reader, which the
    unit tests above only exercise a link at a time. Row content is asserted against the counting
    filesystem instead: _reconstruct_path_from_base_url strips the leading slash off a url whose
    scheme carries no authority, so the partitioned branch addresses file:// paths relative to the
    working directory and matches nothing.
    """
    rows = [
        {
            "timestamp": _COUNTING_TREE_END - timedelta(hours=offset),
            "value": offset,
        }
        for offset in range(48)
    ]
    df = pd.DataFrame(rows)
    for unit in ["year", "month", "day", "hour"]:
        df[unit] = getattr(pd.DatetimeIndex(df["timestamp"]), unit)

    with tempfile.TemporaryDirectory() as temp_dir:
        df.to_parquet(
            temp_dir, partition_cols=["year", "month", "day", "hour"], engine="pyarrow"
        )

        def _fail(*args, **kwargs):
            raise AssertionError(
                "find_partitions must not be called for a declared layout"
            )

        monkeypatch.setattr(storey.utils, "find_partitions", _fail)

        target = mlrun.datastore.targets.ParquetTarget(
            path=f"file://{temp_dir}", time_partitioning_granularity="hour"
        )
        result = target.as_df(
            start_time=_COUNTING_START,
            end_time=_COUNTING_END,
            time_column="timestamp",
        )

    # the declared time units are partition columns, not data
    assert "hour" not in result.columns


def test_unpartitioned_target_reads_back_what_the_write_path_wrote(rundb_mock):
    """A granularity declared alongside partitioned=False is written flat and must read back."""
    rows = [
        {"key": str(offset), "timestamp": _COUNTING_TREE_END - timedelta(hours=offset)}
        for offset in range(48)
    ]
    feature_set = mlrun.feature_store.FeatureSet(
        name="unpartitioned",
        entities=[mlrun.feature_store.Entity("key")],
        timestamp_key="timestamp",
    )
    feature_set._run_db = rundb_mock
    feature_set.reload = Mock()
    feature_set.save = Mock()
    feature_set.purge_targets = Mock()

    with tempfile.TemporaryDirectory() as temp_dir:
        target = mlrun.datastore.targets.ParquetTarget(
            path=temp_dir, partitioned=False, time_partitioning_granularity="hour"
        )
        feature_set.ingest(source=pd.DataFrame(rows), targets=[target])

        assert not any(name.startswith("year=") for name in os.listdir(temp_dir))

        result = target.as_df(
            start_time=_COUNTING_START,
            end_time=_COUNTING_END,
            time_column="timestamp",
        )

    assert len(result) == 1


@pytest.mark.parametrize(
    "description, declared",
    [
        ("the pruned walk lists a path carrying no partition values", _HOUR_LAYOUT),
        ("pyarrow filters on levels that are not columns", ["year", "month", "day"]),
    ],
)
def test_a_wrong_declared_layout_reads_what_discovery_reads(
    counting_fs, description, declared
):
    """A layout the store does not have returns the rows discovery returns, not an error."""
    _write_flat_counting_tree(counting_fs, hours=24)

    expected = _read_counting_tree(counting_fs)
    result = _read_counting_tree(counting_fs, partitions=declared)

    pd.testing.assert_frame_equal(result, expected)
    assert len(result) == 1, description


def test_a_wrong_declared_layout_is_reported(counting_fs, monkeypatch):
    """The mismatch surfaces as a warning naming both layouts rather than being absorbed."""
    _write_flat_counting_tree(counting_fs, hours=24)
    warn_spy = MagicMock()
    monkeypatch.setattr(mlrun.utils.logger, "warning", warn_spy)

    _read_counting_tree(counting_fs, partitions=_HOUR_LAYOUT)

    warn_spy.assert_called_once()
    args, kwargs = warn_spy.call_args
    assert "partition layout" in args[0]
    assert kwargs["url"] == _COUNTING_BASE
    assert kwargs["declared"] == _HOUR_LAYOUT
    assert kwargs["discovered"] == []


def test_a_matching_declared_layout_does_not_fall_back(counting_fs, monkeypatch):
    """The optimization is intact: a layout the store has is never rediscovered."""
    _write_counting_tree(counting_fs, hours=24)

    def _fail(*args, **kwargs):
        raise AssertionError("find_partitions must not be called for a matching layout")

    monkeypatch.setattr(storey.utils, "find_partitions", _fail)
    warn_spy = MagicMock()
    monkeypatch.setattr(mlrun.utils.logger, "warning", warn_spy)

    result = _read_counting_tree(counting_fs, partitions=_HOUR_LAYOUT)

    assert len(result) == 1
    warn_spy.assert_not_called()


def test_the_fallback_does_not_loop(counting_fs, monkeypatch):
    """A read still failing after rediscovery raises instead of rediscovering again."""
    _write_flat_counting_tree(counting_fs, hours=24)
    discoveries = []

    def _still_wrong(url, fs):
        discoveries.append(url)
        return ["year", "month", "day"], ["year", "month", "day"]

    monkeypatch.setattr(storey.utils, "find_partitions", _still_wrong)

    with pytest.raises(pyarrow.lib.ArrowInvalid, match="No match for FieldRef"):
        _read_counting_tree(counting_fs, partitions=_HOUR_LAYOUT)

    assert discoveries == [_COUNTING_BASE]


def test_a_filter_on_a_column_that_is_not_a_level_is_not_a_mismatch(
    counting_fs, monkeypatch
):
    """A caller's filter over a missing column raises as it did, without a discovery walk."""
    _write_counting_tree(counting_fs, hours=24)

    def _fail(*args, **kwargs):
        raise AssertionError("find_partitions must not be called for a filter error")

    monkeypatch.setattr(storey.utils, "find_partitions", _fail)

    reader = mlrun.datastore.base.DataStore._parquet_reader(
        pd,
        _COUNTING_BASE,
        counting_fs,
        "timestamp",
        _COUNTING_START,
        _COUNTING_END,
        [("no_such_column", "=", 1)],
        True,
        _HOUR_LAYOUT,
    )

    with pytest.raises(pyarrow.lib.ArrowInvalid, match="no_such_column"):
        reader(_COUNTING_BASE)


@pytest.mark.parametrize(
    "description, exception, expected",
    [
        (
            "the pruned walk reached a path with no partition values",
            TypeError(
                "DataStore._is_directory_in_range() missing 1 required positional "
                "argument: 'year'"
            ),
            True,
        ),
        (
            "pyarrow rejected a filter on a declared level",
            pyarrow.lib.ArrowInvalid(
                "No match for FieldRef.Name(year) in timestamp: timestamp[ns]"
            ),
            True,
        ),
        (
            "pyarrow rejected a filter on a column nobody declared",
            pyarrow.lib.ArrowInvalid(
                "No match for FieldRef.Name(no_such_column) in timestamp: timestamp[ns]"
            ),
            False,
        ),
        (
            "the file is not a parquet file",
            pyarrow.lib.ArrowInvalid(
                "Error creating dataset. Could not read schema from 'year=2026/x.parquet'. "
                "Is this a 'parquet' file?: Parquet magic bytes not found in footer."
            ),
            False,
        ),
    ],
)
def test_only_a_layout_error_is_a_layout_mismatch(description, exception, expected):
    """The discriminator separates a wrong layout from a corrupt file or a caller's filter."""
    assert (
        mlrun.datastore.base.DataStore._is_partition_layout_mismatch(
            exception, _HOUR_LAYOUT
        )
        is expected
    ), description


def test_a_layout_declared_by_a_target_the_write_path_ignored_still_reads(rundb_mock):
    """A feature set with no timestamp key is written flat and must read back what it wrote."""
    rows = [
        {"key": str(offset), "timestamp": _COUNTING_TREE_END - timedelta(hours=offset)}
        for offset in range(48)
    ]
    feature_set = mlrun.feature_store.FeatureSet(
        name="declared-but-flat",
        entities=[mlrun.feature_store.Entity("key")],
        engine="pandas",
    )
    feature_set._run_db = rundb_mock
    feature_set.reload = Mock()
    feature_set.save = Mock()
    feature_set.purge_targets = Mock()

    with tempfile.TemporaryDirectory() as temp_dir:
        target = mlrun.datastore.targets.ParquetTarget(
            path=temp_dir, partitioned=True, time_partitioning_granularity="hour"
        )
        feature_set.ingest(source=pd.DataFrame(rows), targets=[target])

        assert not any(name.startswith("year=") for name in os.listdir(temp_dir))

        result = target.as_df(
            start_time=_COUNTING_START,
            end_time=_COUNTING_END,
            time_column="timestamp",
        )

    assert len(result) == 1
