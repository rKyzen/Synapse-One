"""Repro: what does the pipeline do with a plain greeting?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tempfile
from synapse.bootstrap import Boot, create_container
from synapse.config.paths import SynapsePaths
from synapse.domain import ModelDescriptor, ModelMetadata, ProviderKind
from synapse.contracts import ModelProvider
from synapse.domain.enums import ProviderState
from synapse.actions import classify_request, requires_workspace_access, extract_workspace_ops
from synapse.planner.heuristic import HeuristicTaskPlanner
from synapse.domain import Capability, ComplexityResult, IntentResult, PrivacyResult, PrivacyMode, Decision


class FakeProvider(ModelProvider):
    provider_id = "fake"
    kind = ProviderKind.LOCAL

    def initialize(self): pass

    def list_models(self):
        return [ModelDescriptor(id="chat-test", provider_id="fake")]

    def to_metadata(self, descriptor):
        return ModelMetadata(id=descriptor.id, provider_id="fake", kind=self.kind,
                             privacy_score=1.0, capabilities={"chat": 1.0})

    def embed(self, texts, *, model=None):
        return [[1.0] for _ in texts]

    def chat(self, request):
        from synapse.domain import ChatResponse
        return ChatResponse(provider_id="fake", model_id="chat-test",
                            kind=self.kind, content="Hi there!", raw={})

    def health(self): return True

    def supports(self, capability): return True

    def shutdown(self): pass


def main():
    home = Path(tempfile.mkdtemp(prefix="syn-repro-"))
    import os
    os.environ["SYNAPSE_HOME"] = str(home)
    paths = SynapsePaths.discover(home)
    boot = Boot(create_container(paths=paths))
    fake = FakeProvider()
    boot.providers._providers["fake"] = fake
    boot.providers._states["fake"] = ProviderState.READY
    boot.registry._models["chat-test"] = ModelMetadata(
        id="chat-test", provider_id="fake", kind=ProviderKind.LOCAL,
        privacy_score=1.0, capabilities={"chat": 1.0},
    )

    prompt = "Hello"
    print("classify_request:", classify_request(prompt).value)
    print("requires_workspace_access:", requires_workspace_access(prompt))
    print("extract_workspace_ops:", extract_workspace_ops(prompt))

    planner = HeuristicTaskPlanner()
    intent = IntentResult(primary=__import__("synapse.domain.enums", fromlist=["IntentType"]).IntentType.GENERAL, confidence=1.0)
    complexity = ComplexityResult(score=10)
    privacy = PrivacyResult(mode=PrivacyMode.BALANCED)
    decision = Decision(can_stay_local=True, internet_required=False, privacy=PrivacyMode.BALANCED,
                        required_capabilities=[Capability.CHAT], preferred_capabilities=[],
                        workspace=__import__("synapse.domain.enums", fromlist=["WorkspaceKind"]).WorkspaceKind.GENERAL)
    dag = planner.plan(prompt, intent, complexity, privacy, decision)
    print("planned tasks:", [(t.kind.value, t.file_output, t.file_hint) for t in dag.tasks])

    project = boot.projects.create_project(name="repro", parent_dir=str(home))
    pid = project.id
    resp = boot.master.process(prompt, workspace=boot.projects.workspace_for(pid),
                               file_operator=boot.projects.file_operator(pid),
                               action_log=boot.projects.action_log(pid),
                               project_id=pid,
                               project_name=project.name,
                               project_path=project.workspace_path,
                               change_panel=boot.projects.change_panel(pid),
                               diagnostics=boot.projects.diagnostics(pid))
    ws = boot.projects.workspace_for(pid)
    print("--- RESPONSE ---")
    print(repr(resp.response[:200]))
    print("workspace files:", ws.list_files())
    ws_path = Path(project.workspace_path)
    print("folders on disk:", sorted(p.relative_to(ws_path).as_posix() for p in ws_path.rglob("*") if p.is_dir()))
    print("files on disk:", sorted(p.relative_to(ws_path).as_posix() for p in ws_path.rglob("*") if p.is_file()))
    print("action log recent:", boot.projects.action_log(pid).recent(3))
    boot.shutdown()


if __name__ == "__main__":
    main()
