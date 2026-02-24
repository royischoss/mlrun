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

import pathlib
from typing import Any, Union

import pytest

import mlrun
from mlrun.serving import GraphContext, QueueStep, V2ModelServer  # noqa
from mlrun.serving.states import ModelRunnerStep, TaskStep

from .demo_states import *  # noqa

try:
    import storey
except Exception:
    pass

engines = [
    "sync",
    "async",
]


def myfunc1(x, context=None):
    assert isinstance(context, GraphContext), "didnt get a valid context"
    return x * 2


def myfunc2(x):
    return x * 2


class Mul(storey.MapClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def do(self, event):
        return event * 2


class ModelTestingClass(V2ModelServer):
    def load(self):
        print("loading")

    def predict(self, request):
        print("predict:", request)
        resp = request["inputs"]
        return resp


def test_basic_flow():
    fn = mlrun.new_function("tests", kind="serving", project="x")
    graph = fn.set_topology("flow", engine="sync")
    graph.add_step(name="s1", class_name="Chain")
    graph.add_step(name="s2", class_name="Chain", after="$prev")
    graph.add_step(name="s3", class_name="Chain", after="$prev")

    server = fn.to_mock_server()
    # graph.plot("flow.png")
    resp = server.test(body=[])
    assert resp == ["s1", "s2", "s3"], "flow1 result is incorrect"

    graph = fn.set_topology("flow", exist_ok=True, engine="sync")
    graph.add_step(name="s2", class_name="Chain")
    graph.add_step(
        name="s1", class_name="Chain", before="s2"
    )  # should place s1 first and s2 after it
    graph.add_step(name="s3", class_name="Chain", after="s2")

    server = fn.to_mock_server()
    resp = server.test(body=[])
    assert resp == ["s1", "s2", "s3"], "flow2 result is incorrect"

    graph = fn.set_topology("flow", exist_ok=True, engine="sync")
    graph.add_step(name="s1", class_name="Chain")
    graph.add_step(name="s3", class_name="Chain", after="$prev")
    graph.add_step(name="s2", class_name="Chain", after="s1", before="s3")

    server = fn.to_mock_server()
    resp = server.test(body=[])
    assert resp == ["s1", "s2", "s3"], "flow3 result is incorrect"
    assert server.context.project == "x", "context.project was not set"


@pytest.mark.parametrize("engine", engines)
def test_handler(engine):
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine=engine)
    graph.to(name="s1", handler="(event + 1)").to(name="s2", handler="json.dumps")
    if engine == "async":
        graph["s2"].respond()

    server = fn.to_mock_server()
    resp = server.test(body=5)
    if engine == "async":
        server.wait_for_completion()
    # the json.dumps converts the 6 to "6" (string)
    assert resp == "6", f"got unexpected result {resp}"


def test_handler_with_context():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine="sync")
    graph.to(name="s1", handler=myfunc1).to(name="s2", handler=myfunc2).to(
        name="s3", handler=myfunc1
    )
    server = fn.to_mock_server()
    resp = server.test(body=5)
    # expect 5 * 2 * 2 * 2 = 40
    assert resp == 40, f"got unexpected result {resp}"


def test_init_class():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine="sync")
    graph.to(name="s1", class_name="Echo").to(name="s2", class_name="RespName")

    server = fn.to_mock_server()
    resp = server.test(body=5)
    assert resp == [5, "s2"], f"got unexpected result {resp}"


def test_step_without_do():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine="sync")
    graph.to(name="bad", class_name="NotAStep")

    server = fn.to_mock_server()
    with pytest.raises(RuntimeError, match="step bad does not have a handler"):
        server.test(body=5)


# ML-11989
def test_step_without_do_async_engine():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine="async")
    graph.to(name="bad", class_name="NotAStep")

    with pytest.raises(
        mlrun.errors.MLRunValueError,
        match="Step 'bad' does not have a handler that can be called",
    ):
        fn.to_mock_server()


def test_on_error():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine="sync")
    graph.add_step(name="s1", class_name="Chain")
    graph.add_step(name="raiser", class_name="Raiser", after="$prev").error_handler(
        name="catch", class_name="EchoError", full_event=True
    )
    graph.add_step(name="s3", class_name="Chain", after="$prev")

    server = fn.to_mock_server()
    resp = server.test(body=[])
    if isinstance(resp, dict):
        assert resp["error"] and resp["origin_state"] == "raiser", "error wasn't caught"
    else:
        assert resp.error and resp.origin_state == "raiser", "error wasn't caught"


def return_type(event):
    return event.__class__.__name__


def test_content_type():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine="sync")
    graph.to(name="totype", handler=return_type)
    server = fn.to_mock_server()

    # test that we json.load() when the content type is json
    resp = server.test(body={"a": 1})
    assert resp == "dict", "invalid type"
    resp = server.test(body="[1,2]")
    assert resp == "list", "did not load json on no type"
    resp = server.test(body={"a": 1}, content_type="application/json")
    assert resp == "dict", "invalid type, should keep dict"
    resp = server.test(body="[1,2]", content_type="application/json")
    assert resp == "list", "did not load json"
    resp = server.test(body="[1,2]", content_type="application/text")
    assert resp == "str", "did not keep as string"
    resp = server.test(body="xx [1,2]")
    assert resp == "str", "did not keep as string"
    resp = server.test(body="xx [1,2]", content_type="application/json", silent=True)
    assert resp.status_code == 400, "did not fail on bad json"

    # test the use of default content type
    fn = mlrun.new_function("tests", kind="serving")
    fn.spec.default_content_type = "application/json"
    graph = fn.set_topology("flow", engine="sync")
    graph.to(name="totype", handler=return_type)

    server = fn.to_mock_server()
    resp = server.test(body="[1,2]")
    assert resp == "list", "did not load json"


def test_add_model():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine="sync")
    graph.to("Echo", "e1").to("Echo", "e2")
    try:
        # should fail, we dont have a router
        fn.add_model("m1", class_name="ModelTestingClass", model_path=".")
        assert True, "add_model did not fail without router"
    except Exception:
        pass

    # model should be added to the one (and only) router
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine="sync")
    graph.to("Echo", "e1").to("*", "router").to("Echo", "e2")
    fn.add_model("m1", class_name="ModelTestingClass", model_path=".")

    assert "m1" in graph["router"].routes, "model was not added to router"

    # model is added to the specified router (by name)
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine="sync")
    graph.to("Echo", "e1").to("*", "r1").to("Echo", "e2").to("*", "r2")
    fn.add_model("m1", class_name="ModelTestingClass", model_path=".", router_step="r2")

    assert "m1" in graph["r2"].routes, "model was not added to proper router"


def test_multi_function():
    # model is added to the specified router (by name)
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine="sync")
    graph.to("Echo", "e1").to("$queue", "q1", path="").to("*", "r1", function="f2").to(
        "Echo", "e2", function="f2"
    )
    fn.add_model("m1", class_name="ModelTestingClass", model_path=".")

    # start from root function
    server = fn.to_mock_server()
    resp = server.test("/v2/models/m1/infer", body={"inputs": [5]})
    server.wait_for_completion()
    print(resp)
    assert resp["outputs"] == [5], "wrong output"

    # start from 2nd function
    server = fn.to_mock_server(current_function="f2")
    resp = server.test(body={"inputs": [5]})
    server.wait_for_completion()
    print(resp)
    assert resp["outputs"] == [5], "wrong output"


path_control_tests = {"handler": (myfunc2, None), "class": (None, "Mul")}


@pytest.mark.parametrize("test_type", path_control_tests.keys())
@pytest.mark.parametrize("engine", engines)
def test_path_control(engine, test_type):
    function = mlrun.new_function("test", kind="serving")
    flow = function.set_topology("flow", engine=engine)

    handler, class_name = path_control_tests[test_type]

    # function input will be event["x"] and result will be written to event["y"]["z"]
    flow.to(
        class_name, handler=handler, name="x2", input_path="x", result_path="y.z"
    ).respond()

    server = function.to_mock_server()
    resp = server.test(body={"x": 5})
    server.wait_for_completion()
    # expect y.z = x * 2 = 10
    assert resp == {"x": 5, "y": {"z": 10}}, "wrong resp"


def test_path_control_routers():
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    graph.to(name="s1", class_name="Echo").to(
        "*", name="r1", input_path="x", result_path="y"
    ).to(name="s3", class_name="Echo").respond()
    function.add_model("m1", class_name="ModelClass", model_path=".")
    server = function.to_mock_server()

    resp = server.test("/v2/models/m1/infer", body={"x": {"inputs": [5]}})
    server.wait_for_completion()
    print(resp)
    assert resp["y"]["outputs"] == 5, "wrong output"

    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="sync")
    graph.to(name="s1", class_name="Echo").to(
        "*mlrun.serving.routers.VotingEnsemble",
        name="r1",
        input_path="x",
        result_path="y",
        vote_type="regression",
    ).to(name="s3", class_name="Echo").respond()
    function.add_model("m1", class_name="ModelClassList", model_path=".", multiplier=10)
    function.add_model("m2", class_name="ModelClassList", model_path=".", multiplier=20)
    server = function.to_mock_server()

    resp = server.test("/v2/models/infer", body={"x": {"inputs": [[5]]}})
    server.wait_for_completion()
    # expect avg of (5*10) and (5*20) = 75
    assert resp["y"]["outputs"] == [75], "wrong output"


def test_to_dict():
    from mlrun.serving.remote import RemoteStep

    rs = RemoteStep(
        name="remote_echo",
        url="/url",
        method="GET",
        input_path="req",
        result_path="resp",
        retries=4,
    )

    assert rs.to_dict() == {
        "name": "remote_echo",
        "class_args": {
            "method": "GET",
            "return_json": True,
            "url": "/url",
            "retries": 4,
        },
        "class_name": "mlrun.serving.remote.RemoteStep",
        "input_path": "req",
        "result_path": "resp",
    }, "unexpected serialization"

    ms = V2ModelServer(name="ms", model_path="./xx", multiplier=7)
    assert ms.to_dict() == {
        "class_args": {"model_path": "./xx", "multiplier": 7, "protocol": "v2"},
        "class_name": "mlrun.serving.v2_serving.V2ModelServer",
        "name": "ms",
    }, "unexpected serialization"


def test_module_load():
    # test that the functions and classes are imported automatically from the function code
    function_path = str(pathlib.Path(__file__).parent / "assets" / "myfunc.py")

    def check_function(name, fn):
        graph = fn.set_topology("flow", engine="sync")
        graph.to(name="s1", class_name="MyCls").to(name="s2", handler="myhand")

        server = fn.to_mock_server()
        resp = server.test(body=5)
        # result should be 5 * 2 * 2 = 20
        assert resp == 20, f"got unexpected result {resp} with {name}"

    check_function(
        "code_to_function",
        mlrun.code_to_function("test1", filename=function_path, kind="serving"),
    )
    check_function(
        "new_function",
        mlrun.new_function("test2", command=function_path, kind="serving"),
    )


def test_missing_functions():
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="async")
    graph.to(name="s1", class_name="Echo").to(
        name="s2", class_name="Echo", function="child_func"
    )
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError, match=r"function child_func*"
    ):
        function.deploy()


def test_add_aggregate_as_insert():
    fn = mlrun.new_function("tests", kind="serving", project="x")
    graph = fn.set_topology("flow", engine="sync")
    graph.add_step(name="s1", class_name="Chain")

    before = "s1"
    after = None
    if before is None and after is None:
        after = "$prev"
    graph.insert_step(
        key="Aggregates",
        step=TaskStep(name="Aggregates", class_name="storey.Aggregates"),
        before=before,
        after=after,
    )

    assert graph["s1"].after == ["Aggregates"]

    graph_2 = fn.set_topology("flow", exist_ok=True, engine="sync")
    graph_2.add_step(name="s1", class_name="Chain").to(name="s2", class_name="Chain")

    before = "s2"
    after = None
    if before is None and after is None:
        after = "$prev"
    graph_2.insert_step(
        key="Aggregates",
        step=TaskStep(name="Aggregates", class_name="storey.Aggregates"),
        before=before,
        after=after,
    )

    assert graph_2["s2"].after == ["Aggregates"]
    assert graph_2["Aggregates"].after == ["s1"]


def test_set_flow_error():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine="sync")
    s1 = dict(name="s1", handler="(event + 1)")
    s2 = dict(name="s2", handler="json.dumps")
    graph.to(**s1).to(**s2)

    r1 = dict(name="r1", handler="(event + 10)")
    r2 = dict(name="r2", handler="json.dumps")
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match=r"set_flow\(\) called on a step that already has downstream steps. "
        "If you want to overwrite existing steps, set force=True.",
    ):
        graph.set_flow(steps=[r1, r2])


def test_set_flow():
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine="sync")
    s1 = dict(name="s1", handler="(event + 1)")
    s2 = dict(name="s2", handler="json.dumps")
    graph.to(**s1).to(**s2)

    r1 = dict(name="r1", handler="(event + 10)")
    r2 = dict(name="r2", handler="json.dumps")
    graph.set_flow(steps=[r1, r2], force=True)

    server = fn.to_mock_server()
    resp = server.test(body=5)
    assert resp == "15"


@pytest.mark.parametrize(
    "steps",
    [
        [
            dict(name="r1", handler="(event + 10)"),
            dict(name="r2", handler="json.dumps"),
        ],
        [
            TaskStep(name="r1", handler="(event + 10)"),
            TaskStep(name="r2", handler="json.dumps"),
        ],
        [
            QueueStep(name="r1", handler="(event + 10)"),
        ],
    ],
)
def test_set_flow_names(
    steps: list[Union[TaskStep, QueueStep, dict[str, Any]]],
):
    fn = mlrun.new_function("tests", kind="serving")
    graph = fn.set_topology("flow", engine="sync")
    graph.set_flow(steps=steps, force=True)
    expected = (
        [step["name"] for step in steps]
        if isinstance(steps[0], dict)
        else [step.name for step in steps]
    )
    assert list(graph.to_dict()["steps"].keys()) == expected


def test_model_runner_with_selector():
    function = mlrun.new_function("tests", kind="serving")
    graph = function.set_topology("flow", engine="sync")
    model_runner_step = ModelRunnerStep(
        name="my_model_runner",
    )
    with pytest.raises(mlrun.serving.states.GraphError):
        graph.to(model_runner_step)

    with pytest.raises(mlrun.serving.states.GraphError):
        graph.add_step(model_runner_step)

    with pytest.raises(mlrun.serving.states.GraphError):
        graph.to(name="s1", handler="(event + 1)").to(model_runner_step)


def test_sync_flow_with_branches():
    fn = mlrun.new_function("tests", kind="serving", project="x")
    graph = fn.set_topology("flow", engine="sync")
    graph.to(name="s1", class_name="Mul")
    graph.to(name="s2", class_name="Mul")
    graph.add_step(name="gather", class_name="Gather", after=["s1", "s2"])
    with pytest.raises(mlrun.serving.states.GraphError):
        fn.to_mock_server()


# ML-11985
def test_mrs_wraps_after():
    after = "other-step"
    assert ModelRunnerStep(name="my_model_runner", after=after).after == [after]


def test_queue_step_function_attribute():
    """Test that QueueStep respects function attribute for child functions."""
    # Test QueueStep has function attribute
    queue = QueueStep(name="test_queue", path="dummy://test", function="child")
    assert queue.function == "child"

    # Test QueueStep function is set via params_to_step
    fn = mlrun.new_function("tests", kind="serving", project="x")
    graph = fn.set_topology("flow", engine="async")
    graph.to(">>", name="q1", path="dummy://test", function="child_func")
    assert graph.steps["q1"].function == "child_func"

    # Test QueueStep _is_local_function logic
    queue_no_func = QueueStep(name="queue_no_func", path="dummy://test")
    queue_with_func = QueueStep(
        name="queue_with_func", path="dummy://test", function="child"
    )

    # Create mock context with current_function
    class MockContext:
        current_function = ""

    context = MockContext()

    # Queue without function should be local when current_function is also empty (parent function)
    assert queue_no_func._is_local_function(context) is True

    # Queue with function="child" should NOT be local when current_function is "" (parent)
    assert queue_with_func._is_local_function(context) is False

    # Queue with function should be local if current_function matches
    context.current_function = "child"
    assert queue_with_func._is_local_function(context) is True

    # Queue without function should NOT be local on child function
    assert queue_no_func._is_local_function(context) is False

    context.current_function = "other"
    assert queue_with_func._is_local_function(context) is False

    context.current_function = "*"
    assert queue_with_func._is_local_function(context) is True
    assert queue_no_func._is_local_function(context) is True


def test_queue_step_with_model_runner_on_child_function():
    """Test that QueueStep after ModelRunnerStep on child function gets initialized correctly."""
    fn = mlrun.new_function("tests", kind="serving", project="x")
    graph = fn.set_topology("flow", engine="async")

    model_runner_step = ModelRunnerStep(name="model_runner", raise_exception=True)
    model_runner_step.add_model(
        model_class="Echo",
        execution_mechanism="naive",
        endpoint_name="my_model",
    )

    # Setup graph: queue -> model_runner on child -> output queue on child
    graph.to(">>", name="input_queue", path="dummy://input").to(
        model_runner_step, function="child"
    ).to(">>", name="output_queue", path="dummy://output", function="child")

    # Verify the output queue has the correct function set
    assert graph.steps["output_queue"].function == "child"
    assert graph.steps["model_runner"].function == "child"


def test_queue_step_on_child_function_receives_messages():
    """Test that QueueStep on child function actually receives messages."""
    fn = mlrun.new_function("tests", kind="serving", project="x")
    graph = fn.set_topology("flow", engine="async")

    # Simple graph: input queue -> process step on child -> output queue on child
    graph.to(">>", name="input_queue", path="dummy://input").to(
        name="process", class_name="Echo", function="child"
    ).to(">>", name="output_queue", path="dummy://output", function="child")

    # Run as the child function
    server = fn.to_mock_server(current_function="child")
    server.test("/", body={"test": "data"})
    server.wait_for_completion()

    # Verify output queue received the message
    output_stream = server.graph.steps["output_queue"].async_object
    assert (
        output_stream is not None
    ), "Output stream should be initialized on child function"
    assert (
        len(output_stream.event_list) == 1
    ), "Output queue should have received one message"
    assert output_stream.event_list[0]["test"] == "data"


def test_queue_step_not_initialized_on_wrong_function():
    """Test that QueueStep with function attribute is not initialized on wrong function."""
    fn = mlrun.new_function("tests", kind="serving", project="x")
    graph = fn.set_topology("flow", engine="async")

    # Graph with local step followed by queue on child function
    # When running as parent, the local step runs, but the child queue should not init stream
    graph.to(name="local_step", class_name="Echo").to(
        ">>", name="child_queue", path="dummy://test", function="child"
    )

    # Run as parent function (empty string means parent)
    server = fn.to_mock_server(current_function="")

    # The queue's _stream should not be set since _is_local_function returns False on parent
    queue_step = server.graph.steps["child_queue"]
    assert (
        queue_step._stream is None
    ), "Queue stream should not be initialized on parent function"
