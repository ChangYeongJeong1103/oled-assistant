"""Run the approved evaluation sequentially, keeping a recovery journal."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace


def runtime_settings(ev):
    """Record effective non-secret settings, rather than copying a dotenv file."""
    names = ('CANDIDATE_TOP_K', 'FINAL_TOP_N', 'MIN_DOC_RELEVANCE',
             'CHUNK_SIZE', 'CHUNK_OVERLAP', 'MODEL_REASONING_EFFORT',
             'MODEL_PRICING_PER_1M')
    settings = {name: value for name, value in vars(ev.config).items()
                if name in names or (name.startswith(('AGENT_', 'SESSION_'))
                                     and not name.endswith('_PATH'))}
    return {'manifest': ev.evaluation_manifest(ev.QUESTIONS_FILE),
            'settings': settings,
            'execution': {'torch_threads': 4, 'agent_question_concurrency': 1,
                          'repeat': 1}}


def main():
    parser = argparse.ArgumentParser(description="Run the 50-question legacy/graph/empty-history comparison and 21-turn development evaluation (paid API calls).")
    parser.add_argument("--run", action="store_true", help="Execute all evaluations using the project models and corpus")
    parser.add_argument("--journal-dir", type=Path, help="Reuse completed answers and judges from this directory")
    parser.add_argument("--env-file", type=Path, help="Optional local dotenv file (never copied to results)")
    args = parser.parse_args()
    if not args.run:
        parser.error("Pass --run to execute the paid evaluation")
    ROOT = Path(__file__).resolve().parents[1]
    OUT = args.journal_dir or ROOT / 'eval/results/raw/upgrade_journal'
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
    from dotenv import load_dotenv
    load_dotenv(args.env_file or ROOT / '.env')

    # Refuse to combine measurements from different source/data revisions.
    paths = sorted([*ROOT.glob('src/*.py'), *ROOT.glob('scripts/*.py'),
                    ROOT / 'eval/questions.json', *ROOT.glob('eval/conversations*.json'),
                    ROOT / 'requirements.txt'])
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths if p.name != 'evaluate_upgrade.py'}
    manifest_path = OUT / 'frozen_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Source or dataset changed; use a new journal directory')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    caches = {}
    for path in OUT.glob('upgrade_*.jsonl'):
        entries = []
        lines = path.read_bytes().splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                entries.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if index != len(lines) - 1 or line.endswith(b'\n'):
                    raise RuntimeError(f'Corrupt journal: {path.name}, line {index + 1}')
                # Keep the interrupted write for inspection, then repair the tail.
                path.with_suffix('.partial').write_bytes(line)
                path.write_bytes(b''.join(lines[:index]))
        if path.stat().st_size and not path.read_bytes().endswith(b'\n'):
            with path.open('ab') as handle:
                handle.write(b'\n')
        caches[path.stem] = entries

    import torch
    torch.set_num_threads(4)
    import evaluate_agent as ev
    from agent_runtime import AgentAssistant
    from agent_graph import GraphAgentAssistant
    from retrieval import build_retriever

    settings = runtime_settings(ev)
    settings_path = OUT / 'runtime_settings.json'
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise RuntimeError('Runtime, corpus or settings changed; use a new journal directory')
    else:
        settings_path.write_text(json.dumps(settings, indent=2) + '\n')

    phase = 'setup'
    lock = threading.Lock()
    consecutive_errors = 0

    def journal(kind, payload):
        with lock, (OUT / (phase + '.jsonl')).open('a') as f:
            f.write(json.dumps({'kind': kind, **payload}, ensure_ascii=False, default=str) + '\n')

    original_run = ev.run_question
    def run_question(agent, item, thread_id=None):
        nonlocal consecutive_errors
        cached = next((entry for entry in caches.get(phase, [])
                       if entry['kind'] == 'agent' and entry['item'] == item), None)
        if cached:
            if phase == 'upgrade_multiturn':
                if 'session' not in cached:
                    raise RuntimeError('Cannot resume a conversation without its saved session')
                # Replay completed conversation context into the evaluator's new isolated thread.
                # Retrieval evidence, budgets and traces are not restored.
                agent.session_graph.update_state(
                    {'configurable': {'thread_id': thread_id}},
                    cached['session'], as_node='finish')
            print('REUSED_AGENT', phase, item['id'], flush=True)
            return deepcopy(cached['record'])
        record = original_run(agent, item, thread_id=thread_id)
        payload = {'item': item, 'record': record}
        if phase == 'upgrade_multiturn':
            payload['session'] = agent.session(thread_id)
        journal('agent', payload)
        consecutive_errors = consecutive_errors + 1 if record['mode'] == 'ERROR' else 0
        if consecutive_errors >= 3:
            raise RuntimeError('Three consecutive operational errors; stopped to investigate')
        return record
    ev.run_question = run_question

    original_judge = ev.judge_answer
    def judge_answer(client, item, record):
        cached = next((entry for entry in caches.get(phase, [])
                       if entry['kind'] == 'judge' and entry['id'] == item['id']
                       and entry['answer'] == record['answer']), None)
        if cached:
            return deepcopy(cached['judgment'])
        result = original_judge(client, item, record)
        journal('judge', {'id': item['id'], 'answer': record['answer'], 'judgment': result})
        return result
    ev.judge_answer = judge_answer

    retriever = build_retriever()
    assert retriever.vectorstore._collection.count() == 7231, 'Unexpected corpus size'
    assert retriever.reranker is not None, 'Reranker must not silently fall back'
    def build_agent(engine='graph', session_db=None):
        if engine == 'legacy':
            return AgentAssistant(retriever)
        return GraphAgentAssistant.with_sqlite(retriever, session_db) if session_db else GraphAgentAssistant(retriever)
    ev.build_agent = build_agent

    print('SETUP_READY', json.dumps(build_agent('legacy').features()), flush=True)
    for phase, engine, session_db in [
        ('upgrade_legacy', 'legacy', None),
        ('upgrade_graph', 'graph', None),
        ('upgrade_empty_history', 'graph', str(OUT / 'empty_history.sqlite')),
    ]:
        print('PHASE_START', phase, flush=True)
        ev.evaluate(SimpleNamespace(questions=None, ids='', engine=engine,
                                   session_db=session_db, no_judge=False,
                                   repeat=1, label=phase))
        print('PHASE_DONE', phase, flush=True)

    phase = 'upgrade_multiturn'
    print('PHASE_START', phase, flush=True)
    import evaluate_multiturn as multi
    original_interpretation = multi.grade_interpretation
    def grade_interpretation(client, turn, record):
        cached = next((entry for entry in caches.get(phase, [])
                       if entry['kind'] == 'interpretation' and entry['id'] == record['id']
                       and entry.get('standalone_question') == record['standalone_question']
                       and entry.get('answer') == record['answer']), None)
        if cached:
            return deepcopy(cached['judgment'])
        result = original_interpretation(client, turn, record)
        journal('interpretation', {'id': record['id'], 'judgment': result,
                                  'standalone_question': record['standalone_question'],
                                  'answer': record['answer']})
        return result
    multi.grade_interpretation = grade_interpretation
    sys.argv = ['evaluate_multiturn.py', '--repeat', '1']
    multi.main()
    print('PHASE_DONE', phase, flush=True)
    print('ALL_EVALUATIONS_DONE', flush=True)


if __name__ == '__main__':
    main()
