"""Tests de `winllama_engine.pipeline`.

No requieren `openvino_genai` instalado: se usa un doble (fake) del modulo
`ov_genai` inyectado via `sys.modules`, de modo que la envoltura se valida
contra la API real de OpenVINO GenAI sin descargar un modelo ni tener GPU.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from winllama_engine import pipeline


class FakeGenerationConfig:
    """Doble de `openvino_genai.GenerationConfig`."""

    def __init__(self) -> None:
        self.max_new_tokens = None
        self.do_sample = None
        self.temperature = None
        self.top_p = None
        self.top_k = None
        self.repetition_penalty = None
        self.stop_strings: set[str] = set()


class FakeStreamingStatus:
    RUNNING = "RUNNING"
    STOP = "STOP"


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True):  # noqa: ANN001
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)


class FakePipeline:
    """Doble de `openvino_genai.LLMPipeline` que verifica la firma real."""

    last_instance: "FakePipeline | None" = None

    def __init__(self, model_dir: str, device: str) -> None:
        self.model_dir = model_dir
        self.device = device
        self.generated: list[tuple[str, object]] = []
        FakePipeline.last_instance = self

    def generate(self, prompt, config=None, streamer=None):  # noqa: ANN001
        self.generated.append((prompt, config))
        text = f"echo:{prompt}"
        if streamer is not None:
            for piece in ("echo:", prompt):
                status = streamer(piece)
                if status != FakeStreamingStatus.RUNNING:
                    break
        else:
            return text

    def get_tokenizer(self) -> FakeTokenizer:
        return FakeTokenizer()


@pytest.fixture()
def fake_ov_genai(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Instala un `openvino_genai` falso en sys.modules."""
    module = types.ModuleType("openvino_genai")
    module.LLMPipeline = FakePipeline  # type: ignore[attr-defined]
    module.GenerationConfig = FakeGenerationConfig  # type: ignore[attr-defined]
    module.StreamingStatus = FakeStreamingStatus  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openvino_genai", module)
    pipeline.clear_cache()
    yield module
    pipeline.clear_cache()


@pytest.fixture()
def ir_model_dir(tmp_path: Path) -> Path:
    """Crea un directorio con un grafo IR ficticio (basta un .xml)."""
    model_dir = tmp_path / "qwen2-0_5b"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    (model_dir / "openvino_model.bin").write_bytes(b"\x00\x01")
    return model_dir


def test_is_model_dir_detects_ir(tmp_path: Path) -> None:
    assert pipeline.is_model_dir(tmp_path) is False
    (tmp_path / "model.xml").write_text("<net/>", encoding="utf-8")
    assert pipeline.is_model_dir(tmp_path) is True


def test_load_raises_when_model_missing(tmp_path: Path, fake_ov_genai) -> None:
    engine = pipeline.LLMEngine(tmp_path / "nope", "CPU")
    with pytest.raises(pipeline.ModelNotFoundError):
        engine.load()


def test_load_uses_real_pipeline_signature(ir_model_dir: Path, fake_ov_genai) -> None:
    engine = pipeline.LLMEngine(ir_model_dir, "CPU")
    engine.load()
    assert engine.loaded is True
    instance = FakePipeline.last_instance
    assert instance is not None
    assert instance.model_dir == str(ir_model_dir)
    assert instance.device == "CPU"


def test_generate_builds_generation_config(ir_model_dir: Path, fake_ov_genai) -> None:
    engine = pipeline.LLMEngine(ir_model_dir, "CPU")
    options = pipeline.GenOptions(max_new_tokens=32, temperature=0.2, stop=["</s>"])
    text = engine.generate("hola", options)
    assert "hola" in text
    _, config = FakePipeline.last_instance.generated[-1]  # type: ignore[union-attr]
    assert config.max_new_tokens == 32
    assert config.temperature == 0.2
    assert config.stop_strings == {"</s>"}


def test_stream_yields_token_chunks(ir_model_dir: Path, fake_ov_genai) -> None:
    engine = pipeline.LLMEngine(ir_model_dir, "CPU")
    chunks = list(engine.stream("mundo"))
    assert chunks == ["echo:", "mundo"]


def test_apply_chat_template_uses_tokenizer(ir_model_dir: Path, fake_ov_genai) -> None:
    engine = pipeline.LLMEngine(ir_model_dir, "CPU")
    prompt = engine.apply_chat_template([{"role": "user", "content": "hola"}])
    assert prompt == "<user>hola</user><assistant>"


def test_gen_options_from_contract_options() -> None:
    options = pipeline.GenOptions.from_contract_options(
        {"num_predict": 10, "temperature": 0.0, "stop": ["\n"]}
    )
    assert options.max_new_tokens == 10
    assert options.temperature == 0.0
    assert options.stop == ["\n"]


# ------------------------------------------------- pruebas T-210 (002 / US1)
# Fuente: data-model.md §4 + ADR-0005 (repetition_penalty=1.0 default greedy)
# + control C-103 (caps num_ctx/num_predict/stop con reject 400).
class TestDefaultsGreedy:
    """T-210: defaults greedy y mapping completo de options del contrato -> GenerationConfig."""

    def test_repetition_penalty_default_is_neutral_1_0(self) -> None:
        """ADR-0005: sin repeat_penalty el motor NO penaliza (ISS-01 cerrado)."""
        assert pipeline.GenOptions().repetition_penalty == 1.0

    def test_from_contract_options_missing_repeat_penalty_is_1_0(self) -> None:
        options = pipeline.GenOptions.from_contract_options({"num_predict": 10})
        assert options.repetition_penalty == 1.0

    def test_from_contract_options_explicit_repeat_penalty_applies(self) -> None:
        options = pipeline.GenOptions.from_contract_options({"repeat_penalty": 1.15})
        assert options.repetition_penalty == 1.15

    def test_blank_request_is_greedy_false(self) -> None:
        """Sin opciones no se muestrea (ADR-0005: solo aplica lo pedido)."""
        options = pipeline.GenOptions.from_contract_options(None)
        assert options.temperature == 0.0
        assert options.is_sampling is False

    def test_temperature_zero_forces_greedy(self) -> None:
        options = pipeline.GenOptions.from_contract_options({"temperature": 0})
        assert options.is_sampling is False

    def test_positive_temperature_is_sampling(self) -> None:
        options = pipeline.GenOptions.from_contract_options({"temperature": 0.8})
        assert options.is_sampling is True
        assert options.temperature == 0.8

    def test_explicit_do_sample_respected(self) -> None:
        options = pipeline.GenOptions.from_contract_options(
            {"do_sample": True, "temperature": 0.2}
        )
        assert options.do_sample is True
        assert options.is_sampling is True
        # do_sample=True + temperature=0 se queda en greedy (sampling degenerado).
        assert (
            pipeline.GenOptions.from_contract_options(
                {"do_sample": True, "temperature": 0.0}
            ).is_sampling
            is False
        )

    def test_seed_num_ctx_min_p_penalties_mapped(self) -> None:
        options = pipeline.GenOptions.from_contract_options(
            {
                "seed": 42,
                "num_ctx": 1024,
                "min_p": 0.05,
                "presence_penalty": 0.4,
                "frequency_penalty": 0.6,
                "temperature": 0.7,
            }
        )
        assert options.seed == 42
        assert options.num_ctx == 1024
        assert options.min_p == 0.05
        assert options.presence_penalty == 0.4
        assert options.frequency_penalty == 0.6

    def test_cap_num_predict_rejects_above_limit(self) -> None:
        """C-103: num_predict > 4096 se rechaza con error explicito."""
        with pytest.raises(pipeline.OptionCapError) as exc:
            pipeline.GenOptions.from_contract_options({"num_predict": 5000})
        assert "num_predict" in str(exc.value)

    def test_cap_num_predict_minus_one_maps_to_cap(self) -> None:
        """El contrato usa num_predict=-1 como 'ilimitado': se clampa al cap."""
        options = pipeline.GenOptions.from_contract_options({"num_predict": -1})
        assert 1 <= options.max_new_tokens <= pipeline.MAX_NUM_PREDICT

    def test_cap_num_ctx_rejects_above_limit(self) -> None:
        with pytest.raises(pipeline.OptionCapError) as exc:
            pipeline.GenOptions.from_contract_options({"num_ctx": 10_000_000})
        assert "num_ctx" in str(exc.value)

    def test_cap_num_predict_at_limit_ok(self) -> None:
        assert pipeline.GenOptions.from_contract_options(
            {"num_predict": pipeline.MAX_NUM_PREDICT}
        ).max_new_tokens == pipeline.MAX_NUM_PREDICT

    def test_cap_stop_rejects_too_many_or_too_long(self) -> None:
        with pytest.raises(pipeline.OptionCapError) as exc:
            pipeline.GenOptions.from_contract_options({"stop": ["sondas"] * 11})
        assert "stop" in str(exc.value)
        with pytest.raises(pipeline.OptionCapError):
            pipeline.GenOptions.from_contract_options({"stop": ["x" * 257]})

    def test_stop_is_list_of_strings(self) -> None:
        """El contrato envia `stop` como array de strings;.nada de slash-strings."""
        options = pipeline.GenOptions.from_contract_options(
            {"stop": ["\n", "user:"]}
        )
        assert options.stop == ["\n", "user:"]

    def test_build_config_greedy(self, ir_model_dir: Path, fake_ov_genai) -> None:
        """En greedy config.do_sample=False y no se aplican knobs de sampling."""
        engine = pipeline.LLMEngine(ir_model_dir, "CPU")
        engine.load()
        options = pipeline.GenOptions.from_contract_options(
            {"temperature": 0, "num_predict": 64, "num_ctx": 2048, "seed": 7}
        )
        engine.generate("hola", options)
        config = FakePipeline.last_instance.generated[-1][1]  # type: ignore[union-attr]
        assert config.do_sample is False
        assert config.max_new_tokens == 64
        assert config.max_length == 2048
        assert config.rng_seed == 7

    def test_build_config_sampling(self, ir_model_dir: Path, fake_ov_genai) -> None:
        engine = pipeline.LLMEngine(ir_model_dir, "CPU")
        engine.load()
        options = pipeline.GenOptions.from_contract_options(
            {
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 50,
                "min_p": 0.05,
                "presence_penalty": 0.3,
                "frequency_penalty": 0.4,
                "repeat_penalty": 1.1,
            }
        )
        engine.generate("hola", options)
        config = FakePipeline.last_instance.generated[-1][1]  # type: ignore[union-attr]
        assert config.do_sample is True
        assert config.temperature == 0.8
        assert config.top_p == 0.95
        assert config.top_k == 50
        assert config.min_p == 0.05
        assert config.presence_penalty == 0.3
        assert config.frequency_penalty == 0.4
        assert config.repetition_penalty == 1.1

    def test_build_config_stop_strings(self, ir_model_dir: Path, fake_ov_genai) -> None:
        engine = pipeline.LLMEngine(ir_model_dir, "CPU")
        engine.load()
        options = pipeline.GenOptions.from_contract_options({"stop": ["\n", "user:"]})
        engine.generate("hola", options)
        config = FakePipeline.last_instance.generated[-1][1]  # type: ignore[union-attr]
        assert config.stop_strings == {"\n", "user:"}


# ------------------------------------------------- pruebas T-211 (002 / US1)
# Contadores de timing en ns (FR-102, P-002): eval_count, eval_duration_ns,
# ttft_ns, load_duration_ns. El cotejo contra `PerfMetrics` NO es realizable
# con la API Python de openvino_genai 2026.4.1.0 (ver ISSUES en bitacora 002):
# con streamer-callback `generate` no expone DecodedResults/perf_metrics;
# el test exige solo coherencia interna de los contadores propios.
class TestTimingCounters:
    def test_generate_exposes_ns_stats(self, ir_model_dir: Path, fake_ov_genai) -> None:
        engine = pipeline.LLMEngine(ir_model_dir, "CPU")
        result = engine.generate("hola", pipeline.GenOptions(max_new_tokens=8))
        assert isinstance(result, str)  # compat Hito 1
        stats = result.stats
        assert stats.eval_count == 2  # el doble emite 2 subwords
        assert stats.eval_duration_ns >= 0
        assert stats.ttft_ns >= 0
        assert stats.load_duration_ns >= 0
        assert stats.total_duration_ns >= stats.eval_duration_ns

    def test_stats_keys_use_contract_names(self, ir_model_dir: Path, fake_ov_genai) -> None:
        engine = pipeline.LLMEngine(ir_model_dir, "CPU")
        result = engine.generate("hola")
        metrics = result.stats.as_contract_metrics()
        assert set(metrics) == {
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        }

    def test_hot_engine_reports_zero_load_duration(
        self, ir_model_dir: Path, fake_ov_genai
    ) -> None:
        engine = pipeline.LLMEngine(ir_model_dir, "CPU")
        engine.generate("primera")  # fria
        result_second = engine.generate("segunda")  # caliente
        assert result_second.stats.load_duration_ns == 0

    def test_stream_yields_incrementally_not_buffered(
        self, monkeypatch: pytest.MonkeyPatch, ir_model_dir: Path, fake_ov_genai
    ) -> None:
        """El primer token sale ANTES de que termine la generacion (ISS-02)."""
        import time as _time

        engine = pipeline.LLMEngine(ir_model_dir, "CPU")
        engine.load()

        real_generate = FakePipeline.generate

        def slow_generate(self, prompt, config=None, streamer=None):  # noqa: ANN001
            self.generated.append((prompt, config))
            status = streamer("echo:")
            assert status == FakeStreamingStatus.RUNNING
            _time.sleep(0.2)  # aqui en 2020 el pipeline 'seguiria generando'
            streamer(prompt)
            return None

        FakePipeline.generate = slow_generate  # type: ignore[method-assign]
        try:
            first_token_seen_at: float | None = None
            done_at: float | None = None

            def first_token_observer() -> None:  # pragma: no cover - no se usa
                return None

            gen = engine.stream("mundo")
            iterator_ = iter(gen)
            first = next(iterator_)
            first_token_seen_at = _time.perf_counter()
            # Mientras el worker estanco dentro del sleep, el consumidor YA
            # recibio el primer token; luego llega el segundo, mas tarde.
            second = next(iterator_)
            done_at = _time.perf_counter()
            assert first == "echo:"
            assert second == "mundo"
            assert done_at - first_token_seen_at >= 0.1
        finally:
            FakePipeline.generate = real_generate  # type: ignore[method-assign]

    def test_cancel_stops_streamer_loop(
        self, ir_model_dir: Path, fake_ov_genai
    ) -> None:
        """cancel() hace que el streamer devuelva STOP y aborte (C-106)."""
        engine = pipeline.LLMEngine(ir_model_dir, "CPU")
        engine.load()
        stream = engine.stream("mundo")
        returned_by_callback: list[object] = []

        real_generate = FakePipeline.generate

        def observing_generate(self, prompt, config=None, streamer=None):  # noqa: ANN001
            self.generated.append((prompt, config))
            import time as _time

            for subword in ("echo:", "mundo"):
                status = streamer(subword)
                returned_by_callback.append(status)
                if status == "STOP":
                    return None
                if subword == "echo:":
                    # El worker espera a que el consumidor cancele antes del
                    # segundo token: garantiza que el STOP se observe.
                    while not stream._abort.is_set():
                        _time.sleep(0.001)
            return None

        FakePipeline.generate = observing_generate  # type: ignore[method-assign]
        try:
            it = iter(stream)
            assert next(it) == "echo:"
            stream.cancel()
            # El callback debe recibir STOP (o al menos cerrarse sin colgar).
            list(it)
            assert returned_by_callback[-1] == "STOP"
        finally:
            FakePipeline.generate = real_generate  # type: ignore[method-assign]


    def test_result_after_explicit_iteration_no_hang(
        self, ir_model_dir: Path, fake_ov_genai
    ) -> None:
        """Regresión daemon: iterar el stream completo y LUEGO result()
        NO debe bloquear esperando un DONE que ya fue consumido (T-223)."""
        engine = pipeline.LLMEngine(ir_model_dir, "CPU")
        stream = engine.stream("mundo")
        tokens = list(stream)
        assert tokens == ["echo:", "mundo"]
        result = stream.result()  # inmediato, sin re-iterar
        assert str(result) == "echo:mundo"
        assert result.stats.eval_count == 2


def test_fallback_chat_prompt() -> None:
    prompt = pipeline._fallback_chat_prompt(
        [{"role": "system", "content": "sos util"}, {"role": "user", "content": "hola"}]
    )
    assert prompt.startswith("System: sos util")
    assert prompt.endswith("Assistant:")


def test_missing_openvino_raises_clear_error(ir_model_dir: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openvino_genai", None)
    pipeline.clear_cache()
    engine = pipeline.LLMEngine(ir_model_dir, "CPU")
    with pytest.raises(pipeline.EngineNotAvailableError):
        engine.load()
    pipeline.clear_cache()
