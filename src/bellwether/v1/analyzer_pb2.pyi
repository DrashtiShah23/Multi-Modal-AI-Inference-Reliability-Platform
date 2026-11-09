import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Exchange(_message.Message):
    __slots__ = ("version", "status", "latency_ms", "response_json", "error")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_JSON_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    version: str
    status: int
    latency_ms: float
    response_json: bytes
    error: str
    def __init__(self, version: _Optional[str] = ..., status: _Optional[int] = ..., latency_ms: _Optional[float] = ..., response_json: _Optional[bytes] = ..., error: _Optional[str] = ...) -> None: ...

class Sample(_message.Message):
    __slots__ = ("trace_id", "traceparent", "model", "received_at", "request_json", "request_error", "primary", "candidate")
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    TRACEPARENT_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    RECEIVED_AT_FIELD_NUMBER: _ClassVar[int]
    REQUEST_JSON_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ERROR_FIELD_NUMBER: _ClassVar[int]
    PRIMARY_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_FIELD_NUMBER: _ClassVar[int]
    trace_id: str
    traceparent: str
    model: str
    received_at: _timestamp_pb2.Timestamp
    request_json: bytes
    request_error: str
    primary: Exchange
    candidate: Exchange
    def __init__(self, trace_id: _Optional[str] = ..., traceparent: _Optional[str] = ..., model: _Optional[str] = ..., received_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., request_json: _Optional[bytes] = ..., request_error: _Optional[str] = ..., primary: _Optional[_Union[Exchange, _Mapping]] = ..., candidate: _Optional[_Union[Exchange, _Mapping]] = ...) -> None: ...

class IngestSamplesRequest(_message.Message):
    __slots__ = ("samples",)
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    samples: _containers.RepeatedCompositeFieldContainer[Sample]
    def __init__(self, samples: _Optional[_Iterable[_Union[Sample, _Mapping]]] = ...) -> None: ...

class IngestSamplesResponse(_message.Message):
    __slots__ = ("accepted", "rejected", "items", "windows_closed")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    REJECTED_FIELD_NUMBER: _ClassVar[int]
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    WINDOWS_CLOSED_FIELD_NUMBER: _ClassVar[int]
    accepted: int
    rejected: int
    items: int
    windows_closed: int
    def __init__(self, accepted: _Optional[int] = ..., rejected: _Optional[int] = ..., items: _Optional[int] = ..., windows_closed: _Optional[int] = ...) -> None: ...

class GetCanaryVerdictRequest(_message.Message):
    __slots__ = ("model", "candidate_version")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_VERSION_FIELD_NUMBER: _ClassVar[int]
    model: str
    candidate_version: str
    def __init__(self, model: _Optional[str] = ..., candidate_version: _Optional[str] = ...) -> None: ...

class CanaryCheck(_message.Message):
    __slots__ = ("name", "value", "threshold", "passed", "sufficient_data", "samples", "detail")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    PASSED_FIELD_NUMBER: _ClassVar[int]
    SUFFICIENT_DATA_FIELD_NUMBER: _ClassVar[int]
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: float
    threshold: float
    passed: bool
    sufficient_data: bool
    samples: int
    detail: str
    def __init__(self, name: _Optional[str] = ..., value: _Optional[float] = ..., threshold: _Optional[float] = ..., passed: _Optional[bool] = ..., sufficient_data: _Optional[bool] = ..., samples: _Optional[int] = ..., detail: _Optional[str] = ...) -> None: ...

class CanaryVerdict(_message.Message):
    __slots__ = ("model", "candidate_version", "decision", "checks")
    class Decision(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        DECISION_UNSPECIFIED: _ClassVar[CanaryVerdict.Decision]
        DECISION_HOLD: _ClassVar[CanaryVerdict.Decision]
        DECISION_PROMOTE: _ClassVar[CanaryVerdict.Decision]
        DECISION_ROLLBACK: _ClassVar[CanaryVerdict.Decision]
    DECISION_UNSPECIFIED: CanaryVerdict.Decision
    DECISION_HOLD: CanaryVerdict.Decision
    DECISION_PROMOTE: CanaryVerdict.Decision
    DECISION_ROLLBACK: CanaryVerdict.Decision
    MODEL_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_VERSION_FIELD_NUMBER: _ClassVar[int]
    DECISION_FIELD_NUMBER: _ClassVar[int]
    CHECKS_FIELD_NUMBER: _ClassVar[int]
    model: str
    candidate_version: str
    decision: CanaryVerdict.Decision
    checks: _containers.RepeatedCompositeFieldContainer[CanaryCheck]
    def __init__(self, model: _Optional[str] = ..., candidate_version: _Optional[str] = ..., decision: _Optional[_Union[CanaryVerdict.Decision, str]] = ..., checks: _Optional[_Iterable[_Union[CanaryCheck, _Mapping]]] = ...) -> None: ...

class Incident(_message.Message):
    __slots__ = ("id", "model", "version", "kind", "severity", "status", "summary", "occurrences", "opened_at", "last_seen_at", "details_json")
    ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    OCCURRENCES_FIELD_NUMBER: _ClassVar[int]
    OPENED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_SEEN_AT_FIELD_NUMBER: _ClassVar[int]
    DETAILS_JSON_FIELD_NUMBER: _ClassVar[int]
    id: int
    model: str
    version: str
    kind: str
    severity: str
    status: str
    summary: str
    occurrences: int
    opened_at: _timestamp_pb2.Timestamp
    last_seen_at: _timestamp_pb2.Timestamp
    details_json: str
    def __init__(self, id: _Optional[int] = ..., model: _Optional[str] = ..., version: _Optional[str] = ..., kind: _Optional[str] = ..., severity: _Optional[str] = ..., status: _Optional[str] = ..., summary: _Optional[str] = ..., occurrences: _Optional[int] = ..., opened_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., last_seen_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., details_json: _Optional[str] = ...) -> None: ...

class ListIncidentsRequest(_message.Message):
    __slots__ = ("model", "status", "limit")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    model: str
    status: str
    limit: int
    def __init__(self, model: _Optional[str] = ..., status: _Optional[str] = ..., limit: _Optional[int] = ...) -> None: ...

class ListIncidentsResponse(_message.Message):
    __slots__ = ("incidents",)
    INCIDENTS_FIELD_NUMBER: _ClassVar[int]
    incidents: _containers.RepeatedCompositeFieldContainer[Incident]
    def __init__(self, incidents: _Optional[_Iterable[_Union[Incident, _Mapping]]] = ...) -> None: ...
