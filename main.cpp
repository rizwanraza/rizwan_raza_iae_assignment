#!/usr/bin/env python3
"""
Simple Multi-Agent Chat System prototype
Implements:
 - Coordinator
 - ResearchAgent
 - AnalysisAgent
 - MemoryAgent (structured memory + in-memory vector store)
 - Simple CLI to run sample scenarios and produce outputs.

Author: Generated for assessment
"""

import time
import json
import math
import os
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter

# ----------------------------
# Utilities
# ----------------------------
def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def simple_tokenize(text: str):
    # Lowercase, split on nonalpha for simplicity
    import re
    tokens = re.findall(r"[a-zA-Z0-9\-]+", text.lower())
    return tokens

def text_to_vector(text: str, vocab: Dict[str,int]=None):
    # simple bag-of-words vector as dict
    tokens = simple_tokenize(text)
    c = Counter(tokens)
    if vocab is None:
        return dict(c)
    else:
        return [c.get(w,0) for w in vocab]

def cosine_similarity_dict(a: Dict[str,int], b: Dict[str,int]) -> float:
    # both are sparse maps token->count
    common = set(a.keys()) & set(b.keys())
    num = sum(a[k]*b[k] for k in common)
    norm_a = math.sqrt(sum(v*v for v in a.values()))
    norm_b = math.sqrt(sum(v*v for v in b.values()))
    if norm_a==0 or norm_b==0:
        return 0.0
    return num / (norm_a*norm_b)

# ----------------------------
# VectorStore (pluggable, in-memory)
# ----------------------------
class VectorStore:
    """
    Minimal in-memory vector store:
     - stores records with embeddings (sparse dicts)
     - supports keyword search and vector similarity search
    """
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add(self, record: Dict[str, Any], embedding: Dict[str,int]):
        rec = dict(record)
        rec['_embedding'] = embedding
        self.records.append(rec)

    def search_by_keyword(self, keyword: str, top_k=5) -> List[Dict[str,Any]]:
        keyword = keyword.lower()
        res = []
        for r in self.records:
            txt = (r.get('title','') + ' ' + r.get('content','')).lower()
            if keyword in txt:
                res.append(r)
        return res[:top_k]

    def similarity_search(self, query_embedding: Dict[str,int], top_k=5) -> List[Tuple[Dict[str,Any], float]]:
        scored = []
        for r in self.records:
            score = cosine_similarity_dict(query_embedding, r['_embedding'])
            scored.append((r, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

# ----------------------------
# MemoryAgent
# ----------------------------
class MemoryAgent:
    """
    Structured memory:
     - conversation_memory: list of messages with timestamps & metadata
     - knowledge_base: vector store with facts/findings
     - agent_state: what each agent learned per task
    """
    def __init__(self):
        self.conversation_memory: List[Dict[str,Any]] = []
        self.kb_store = VectorStore()
        self.agent_state: Dict[str, List[Dict[str,Any]]] = defaultdict(list)

    # Conversation Memory
    def add_conversation(self, role: str, text: str, metadata: Dict[str,Any]=None):
        rec = dict(
            ts=now_ts(),
            role=role,
            text=text,
            metadata=metadata or {}
        )
        self.conversation_memory.append(rec)
        return rec

    def search_conversation(self, keyword: str):
        keyword = keyword.lower()
        return [m for m in self.conversation_memory if keyword in m['text'].lower()]

    # Knowledge Base
    def add_kb(self, title: str, content: str, source: str, agent: str, topic: List[str], confidence: float):
        embedding = Counter(simple_tokenize(title + " " + content))
        record = {
            'id': f'kb-{len(self.kb_store.records)+1}',
            'title': title,
            'content': content,
            'source': source,
            'agent': agent,
            'topics': topic,
            'confidence': confidence,
            'ts': now_ts()
        }
        self.kb_store.add(record, embedding)
        return record

    def kb_search_keyword(self, keyword: str, top_k=5):
        return self.kb_store.search_by_keyword(keyword, top_k=top_k)

    def kb_similarity(self, text: str, top_k=5):
        emb = Counter(simple_tokenize(text))
        return self.kb_store.similarity_search(emb, top_k=top_k)

    # Agent state
    def update_agent_state(self, agent_name: str, task_id: str, note: str, result_summary: str):
        rec = {
            'ts': now_ts(),
            'task_id': task_id,
            'note': note,
            'summary': result_summary
        }
        self.agent_state[agent_name].append(rec)
        return rec

# ----------------------------
# ResearchAgent
# ----------------------------
class ResearchAgent:
    """
    Simulated information retrieval using a pre-loaded knowledge base (mock web search).
    For demo purposes, we use an internal mini KB loaded at init.
    """
    def __init__(self, memory: MemoryAgent):
        self.memory = memory
        self.local_kb = self._load_mock_kb()
        # load local_kb entries into memory.kb so MemoryAgent has provenance
        for entry in self.local_kb:
            self.memory.add_kb(
                title=entry['title'],
                content=entry['content'],
                source=entry['source'],
                agent='ResearchAgent (seed)',
                topic=entry['topics'],
                confidence=0.9
            )

    def _load_mock_kb(self):
        # Minimal set of documents to satisfy the test scenarios
        docs = [
            {
                'title': 'Types of Neural Networks',
                'content': 'Feedforward networks, Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), LSTM, GRU, Transformers, Graph Neural Networks (GNNs).',
                'source': 'mock_kb',
                'topics': ['neural networks', 'ml']
            },
            {
                'title': 'Transformer architecture overview',
                'content': 'Transformers use self-attention; components: multi-head attention, positional encodings, feedforward layers. Efficiency: attention is O(n^2) in sequence length; variants exist (sparse, linearized).',
                'source': 'mock_kb',
                'topics': ['transformer', 'architecture', 'efficiency']
            },
            {
                'title': 'Optimizer comparison',
                'content': 'Gradient descent, SGD with momentum, AdaGrad, RMSProp, Adam. Adam typically converges faster on noisy problems; classical GD can be stable for convex problems. Tradeoffs include memory and tuning hyperparameters.',
                'source': 'mock_kb',
                'topics': ['optimizer', 'training']
            },
            {
                'title': 'Recent reinforcement learning papers (mock list)',
                'content': 'PaperA: policy gradients variant; PaperB: off-policy actor-critic; common challenges: sample efficiency, exploration, reproducibility.',
                'source': 'mock_kb',
                'topics': ['reinforcement learning', 'papers', 'rl']
            }
        ]
        return docs

    def find(self, query: str, top_k=5):
        # Simple matching: keyword search + similarity to memory
        tokens = simple_tokenize(query)
        results = []
        # search memory KB first for relevant records (simulate web)
        sim = self.memory.kb_similarity(query, top_k=top_k)
        for rec, score in sim:
            results.append({
                'title': rec['title'],
                'content': rec['content'],
                'source': rec['source'],
                'score': score,
                'provenance': rec
            })
        # If results are weak, add fallback from internal local_kb
        if not results or all(r['score'] < 0.05 for r in results):
            for entry in self.local_kb:
                if any(t in entry['title'].lower() or t in entry['content'].lower() for t in tokens):
                    results.append({
                        'title': entry['title'],
                        'content': entry['content'],
                        'source': entry['source'],
                        'score': 0.5,
                        'provenance': entry
                    })
        # Attach confidence heuristic
        for r in results:
            r['confidence'] = min(0.95, 0.5 + r.get('score',0))
        return results[:top_k]

# ----------------------------
# AnalysisAgent
# ----------------------------
class AnalysisAgent:
    """
    Performs comparisons, simple reasoning, and simple calculations on research outputs.
    Returns structured analysis and a confidence score.
    """
    def __init__(self, memory: MemoryAgent):
        self.memory = memory

    def compare_optimizers(self, items: List[Dict[str,str]]):
        # Very simple rule-based comparison using keywords
        findings = []
        for it in items:
            text = it['content'].lower()
            score = 0.5
            if 'adam' in text:
                score += 0.2
            if 'converge' in text or 'converges' in text or 'faster' in text:
                score += 0.1
            findings.append({
                'item': it['title'],
                'summary': it['content'][:300],
                'score': min(1.0, score)
            })
        # simple ranking by score
        findings.sort(key=lambda x: x['score'], reverse=True)
        return {
            'analysis': findings,
            'confidence': 0.75
        }

    def analyze_transformer_efficiency(self, research_hits: List[Dict[str,Any]]):
        # Aggregate notes and compute a simple cost metric from tokens:
        notes = []
        for h in research_hits:
            notes.append(h['content'])
        combined = "\n".join(notes)
        # find mentions of O(n^2)
        cost_note = 'attention O(n^2) mentioned' if 'o(n^2)' in combined.lower() or 'o(n2)' in combined.lower() or 'n^2' in combined.lower() else 'no explicit quadratic mention'
        summary = f"Combined notes: {combined[:400]} ... | {cost_note}"
        conf = 0.7 if 'o(n^2)' in combined.lower() or 'n^2' in combined.lower() else 0.5
        return {
            'summary': summary,
            'confidence': conf
        }

    def compare_approaches(self, a_desc: str, b_desc: str):
        # naive comparison: count keywords and produce recommendation.
        a_tokens = simple_tokenize(a_desc)
        b_tokens = simple_tokenize(b_desc)
        a_complex = sum(1 for t in a_tokens if len(t) > 6)
        b_complex = sum(1 for t in b_tokens if len(t) > 6)
        recommendation = "Approach A" if a_complex <= b_complex else "Approach B"
        detail = {
            'A_complexity_score': a_complex,
            'B_complexity_score': b_complex,
            'recommendation': recommendation,
            'rationale': f"Lower token-length complexity preferred for resource-constrained scenario."
        }
        return {
            'detail': detail,
            'confidence': 0.6 + 0.1 * (abs(a_complex - b_complex) / max(1, (a_complex+b_complex)))
        }

# ----------------------------
# Coordinator
# ----------------------------
class Coordinator:
    """
    Orchestrates the agents:
     - receives user query
     - performs simple complexity analysis (keywords + length)
     - plans which agents are needed and in what order
     - merges results, updates memory, handles fallback
    """
    def __init__(self):
        self.memory = MemoryAgent()
        self.research = ResearchAgent(self.memory)
        self.analysis = AnalysisAgent(self.memory)
        self.trace: List[Dict[str,Any]] = []

    def log(self, role: str, message: str, payload: Dict[str,Any]=None):
        entry = {
            'ts': now_ts(),
            'role': role,
            'message': message,
            'payload': payload or {}
        }
        self.trace.append(entry)
        print(f"[{entry['ts']}] {role}: {message}")
        if payload:
            print("  payload:", json.dumps(payload, default=str)[:1000])

    def complexity_analysis(self, query: str) -> Dict[str,Any]:
        tokens = simple_tokenize(query)
        plan = {
            'requires_research': False,
            'requires_analysis': False,
            'requires_memory_lookup': False,
            'notes': []
        }
        # heuristics
        research_keywords = ['research','find','papers','recent','summarize','what are','list','describe','show']
        analysis_keywords = ['analyze','compare','which','best','effectiveness','trade-off','tradeoffs','efficiency','recommend']
        memory_keywords = ['what did we discuss','earlier','remember','what did we learn','previous','before']

        qlow = query.lower()
        if any(k in qlow for k in research_keywords) or len(tokens) > 6:
            plan['requires_research'] = True
        if any(k in qlow for k in analysis_keywords):
            plan['requires_analysis'] = True
        if any(k in qlow for k in memory_keywords):
            plan['requires_memory_lookup'] = True

        # fallback: always do a KB similarity check to avoid redundant work
        plan['similarity_check'] = True
        return plan

    def handle_query(self, user_query: str) -> Dict[str,Any]:
        task_id = f"task-{int(time.time())}"
        self.log('Coordinator', f"Received query (task_id={task_id})", {'query': user_query})
        self.memory.add_conversation('user', user_query, {'task_id': task_id})
        plan = self.complexity_analysis(user_query)
        self.log('Coordinator', 'Planned steps', plan)

        final_answer = []
        overall_confidence = 0.0

        # Step 0: memory lookup to see if prior knowledge exists
        if plan.get('similarity_check'):
            sim_hits = self.memory.kb_similarity(user_query, top_k=3)
            self.log('Coordinator', 'Memory similarity search results', {'count': len(sim_hits)})
            # If high similarity (>0.6), reuse memory and skip heavy work
            if sim_hits and sim_hits[0][1] > 0.6:
                rec, score = sim_hits[0]
                summary = f"Found prior knowledge: {rec['title']} (confidence {rec.get('confidence',0.0)})."
                self.log('Coordinator', 'Reusing memory result', {'title': rec['title'], 'score': score})
                self.memory.add_conversation('system', f"Reused memory: {rec['title']}", {'task_id': task_id})
                return {
                    'task_id': task_id,
                    'answer': rec['content'],
                    'trace': self.trace,
                    'source': 'memory',
                    'confidence': rec.get('confidence', 0.8)
                }

        # Step 1: Research (if required)
        research_results = []
        if plan['requires_research']:
            self.log('Coordinator', 'Invoking ResearchAgent', {'query': user_query})
            research_results = self.research.find(user_query, top_k=5)
            self.log('ResearchAgent', 'Returned results', {'count': len(research_results)})
            # record what research agent learned
            self.memory.update_agent_state('ResearchAgent', task_id, 'retrieved documents', 
                                           ', '.join([r['title'] for r in research_results][:5]))
            # store top findings into KB (simulate learning)
            for r in research_results[:2]:
                self.memory.add_kb(title=r['title'], content=r['content'], source=r['source'],
                                   agent='ResearchAgent', topic=[t for t in r.get('provenance',{}).get('topics',[])], confidence=r.get('confidence',0.6))
            overall_confidence = max(overall_confidence, max((r.get('confidence',0.5) for r in research_results), default=0.0))

        # Step 2: Analysis (if required)
        analysis_result = None
        if plan['requires_analysis']:
            self.log('Coordinator', 'Invoking AnalysisAgent', {'depends_on': 'research_results', 'research_count': len(research_results)})
            if 'transformer' in user_query.lower():
                analysis_result = self.analysis.analyze_transformer_efficiency(research_results)
            elif 'optimizer' in user_query.lower() or 'optim' in user_query.lower():
                analysis_result = self.analysis.compare_optimizers(research_results)
            elif 'compare' in user_query.lower():
                # try to parse two approaches from query naively: split by 'and' or 'vs'
                parts = None
                q = user_query.lower()
                if ' vs ' in q:
                    parts = q.split(' vs ')
                elif ' and ' in q:
                    parts = q.split(' and ')
                if parts and len(parts) >= 2:
                    a_desc = parts[0]
                    b_desc = parts[1]
                    analysis_result = self.analysis.compare_approaches(a_desc, b_desc)
                else:
                    analysis_result = {'analysis': [], 'confidence': 0.4}
            else:
                analysis_result = {'analysis': [], 'confidence': 0.5}
            self.log('AnalysisAgent', 'Analysis complete', {'confidence': analysis_result.get('confidence', 0.5)})
            self.memory.update_agent_state('AnalysisAgent', task_id, 'performed analysis', json.dumps(analysis_result)[:400])
            overall_confidence = max(overall_confidence, analysis_result.get('confidence', 0.0))

        # Step 3: Synthesize answer
        answer_parts = []
        if research_results:
            answer_parts.append("Research findings:\n" + "\n".join([f"- {r['title']}: {r['content']}" for r in research_results[:3]]))
        if analysis_result:
            if 'analysis' in analysis_result:
                answer_parts.append("Analysis summary:\n" + json.dumps(analysis_result['analysis'], indent=2)[:1000])
            elif 'summary' in analysis_result:
                answer_parts.append("Analysis summary:\n" + analysis_result['summary'])
            elif 'detail' in analysis_result:
                answer_parts.append("Comparison result:\n" + json.dumps(analysis_result['detail'], indent=2))
        if not answer_parts:
            # Try memory keyword search
            kw = user_query.split()[0]
            mem_hits = self.memory.search_conversation(kw)
            if mem_hits:
                answer_parts.append("Found earlier conversation pieces: " + "; ".join([m['text'] for m in mem_hits[:3]]))
            else:
                answer_parts.append("I could not find relevant knowledge. Try rephrasing or ask for a simpler question.")
            overall_confidence = max(overall_confidence, 0.2)

        final_answer_text = "\n\n".join(answer_parts)
        # Update memory with final result
        self.memory.add_conversation('system', final_answer_text, {'task_id': task_id, 'confidence': overall_confidence})
        # Persist synthesized finding into KB for future reuse
        self.memory.add_kb(title=f"Synthesis for {task_id}", content=final_answer_text, source='Coordinator', agent='Coordinator', topic=simple_tokenize(user_query)[:5], confidence=round(overall_confidence,2))

        self.log('Coordinator', 'Task complete', {'task_id': task_id, 'confidence': overall_confidence})
        return {
            'task_id': task_id,
            'answer': final_answer_text,
            'trace': self.trace,
            'source': 'synthesized',
            'confidence': overall_confidence
        }

# ----------------------------
# Tests / Scenarios
# ----------------------------
def run_scenarios(coordinator: Coordinator, out_dir: str):
    scenarios = {
        'simple_query': "What are the main types of neural networks?",
        'complex_query': "Research transformer architectures, analyze their computational efficiency, and summarize key trade-offs.",
        'memory_test': "What did we discuss about neural networks earlier?",
        'multi_step': "Find recent papers on reinforcement learning, analyze their methodologies, and identify common challenges.",
        'collaborative': "Compare convolutional neural networks and transformers and recommend which is better for image classification."
    }
    os.makedirs(out_dir, exist_ok=True)
    outputs = {}
    for name, q in scenarios.items():
        print("\n" + "="*20 + f" Running scenario: {name} " + "="*20)
        result = coordinator.handle_query(q)
        text = f"--- SCENARIO: {name} ---\nQuery: {q}\n\nAnswer:\n{result['answer']}\n\nConfidence: {result['confidence']}\n\nTrace:\n"
        for t in result['trace'][-8:]:
            text += f"{t['ts']} - {t['role']}: {t['message']}\n"
        # save
        out_path = os.path.join(out_dir, f"{name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(text)
        outputs[name] = text
    return outputs

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    coordinator = Coordinator()
    outputs = run_scenarios(coordinator, out_dir="outputs")
    print("\nAll scenario outputs saved to ./outputs/")
