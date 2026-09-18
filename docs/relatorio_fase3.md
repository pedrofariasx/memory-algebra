# Relatório Fase 3: Diferenciação da Álgebra vs. Baselines

**Data:** 2026-09-15 (rev. 2026-09-17)
**Dados brutos:** `experiments/results_phase3.json`, `results_phase3_honest.json`, `results_phase3_advantage.json`
**Scripts:** `run_phase3_diferenciacao.py`, `run_phase3_honest_baselines.py`, `run_phase3_algebra_advantage.py`

---

## Sumário Executivo

A Fase 3 investiga onde a álgebra $(V, G, T, P)$ se diferencia de sistemas de memória existentes. Após revisão crítica, o experimento original (3a) foi identificado como metodologicamente fraco: o kNN baseline não possuía timestamps, delete nem arestas — capacidades triviais em qualquer vector DB de produção.

**Resultados revisados:**

| Experimento | Álgebra vs. baselines competentes |
|:------------|:----------------------------------|
| 3b: Baselines honestos (temporal + delete + arestas) | **Empate** (100% vs 100%) |
| 3c: Travessia híbrida (aresta + similaridade + aresta) | **Álgebra vence** (100% vs 23-63%) |
| 3c: Composição multi-fonte (IDs diferentes, quotient funde) | **Álgebra vence** (100% vs 23-37%) |
| 3c: Clustering implícito (sem arestas) | Empate (100% vs 100%) |

**Conclusão:** a álgebra não é superior em tarefas pontuais onde baselines recebem capacidades equivalentes. A vantagem estrutural está na **travessia híbrida** (alternar arestas explícitas com pontes de similaridade em uma única operação de hops) e na **fusão automática de entidades com IDs diferentes** via quotient — cenários que nenhum sistema isolado (pure KG ou pure vector DB) resolve sem infraestrutura adicional.

---

## 3a: Comparação Original (histórico — metodologia fraca)

> **Nota:** Este experimento comparou contra um kNN sem timestamps, sem delete e sem arestas. Os resultados (100% vs 50%/33%/87%) são artefatos do design do baseline, não achados empíricos genuínos. Mantido apenas como registro histórico.

| Cenário | Álgebra | kNN naive |
|:--------|:--------|:----------|
| Conflito temporal | 100% | 50% |
| Retração | 100% | 33-43% |
| Multi-hop | 100% | 87-93% |

---

## 3b: Baselines Honestos

Baselines com capacidades equivalentes:
- **TemporalVectorStore:** busca vetorial + filtro por timestamp + delete (equivalente a Qdrant/Weaviate com metadata filtering)
- **SimpleKG:** triplas (sujeito, relação, objeto) + busca por hop + delete

| Cenário | Álgebra | TVS | KG |
|:--------|:--------|:----|:---|
| Conflito temporal ($\rho=0.0$) | 100% | 100% | 50% |
| Conflito temporal ($\rho=0.8$) | 100% | 100% | 50% |
| Retração ($\rho=0.0$) | 100% | 100% | 100% |
| Retração ($\rho=0.8$) | 100% | 100% | 100% |
| Multi-hop ($\rho=0.0$) | 100% | — | 100% |
| Multi-hop ($\rho=0.8$) | 100% | — | 100% |

**Conclusão 3b:** com capacidades equivalentes, empate. A álgebra não é superior em tarefas que baselines resolvem nativamente.

---

## 3c: Vantagem Estrutural

Cenários onde a arquitetura da álgebra (quotient + hops sobre clusters) resolve o que nenhum sistema isolado consegue:

### Travessia Híbrida

Cadeia: A →(aresta)→ B ~~(similaridade, sem aresta)~~ C →(aresta)→ D

| Sistema | $\rho=0.0$ | $\rho=0.8$ |
|:--------|:-----------|:-----------|
| Álgebra (hops sobre quotient) | **100%** | **100%** |
| SimpleKG (só arestas) | 57% | 63% |
| VectorStore (só cosine) | 23% | 43% |

O KG não consegue atravessar B→C (sem aresta). O VectorStore encontra B por similaridade mas não segue para C/D.

### Composição Multi-fonte

3 agentes contribuem fragmentos de uma cadeia A→B→C→D usando IDs diferentes para os mesmos conceitos (ag1:B ≠ ag2:B). O quotient funde por similaridade cosine ≥ θ.

| Sistema | $\rho=0.0$ | $\rho=0.8$ |
|:--------|:-----------|:-----------|
| Álgebra | **100%** | **100%** |
| SimpleKG (sem entity resolution) | 23% | 37% |

### Clustering Implícito

Fatos semanticamente relacionados sem arestas explícitas. KG com busca cosine top-k também agrupa corretamente.

| Sistema | $\rho=0.0$ | $\rho=0.8$ |
|:--------|:-----------|:-----------|
| Álgebra | 100% | 100% |
| SimpleKG (busca cosine) | 100% | 100% |

---

## 3d: Análise de Chaining (limiar adaptativo)

**Script:** `experiments/run_chaining_analysis.py`
**Dados brutos:** `experiments/results_chaining.json`

O fecho transitivo single-linkage do quotient pode encadear nós semanticamente distintos em um único componente ("chaining") quando o limiar θ é baixo demais para a distribuição de similaridade dos embeddings.

### Embeddings reais (all-MiniLM-L6-v2, 200 frases, 20 tópicos)

Distribuição de similaridade par-a-par: μ=0.358, σ=0.173.

| θ | Componentes | Maior cluster | Observação |
|:--|:------------|:--------------|:-----------|
| 0.50 | 4 | 160/200 | chaining severo |
| 0.60 | 13 | 40 | chaining parcial |
| 0.70 | 18 | 20 | estrutura de tópicos recuperada |
| 0.80 | 19 | 20 | estrutura de tópicos recuperada |
| 0.85 | 20 | 10 | estrutura ideal |
| 0.90 | 21 | 10 | estrutura ideal |
| **Adaptativo (μ+2σ=0.70)** | **18** | **20** | **heurística válida** |

### Conclusões

1. **Chaining é real e quantificável** em embeddings reais com θ baixo (θ=0.5 colapsa 80% dos nós em 1 componente).
2. **θ=0.85 (default do sistema) evita chaining completamente** — 20 componentes para 20 tópicos.
3. **θ adaptativo (μ+2σ)** é uma heurística válida que se aproxima do θ ideal sem tuning manual, útil para deploy com distribuições desconhecidas.
4. Para vetores sintéticos ortogonais, chaining é inexistente em qualquer θ (similaridade média ≈ 0).

---

## Posicionamento Corrigido para o Paper

A contribuição empírica da álgebra **não** é "superior em tarefas temporais/retração/multi-hop" (baselines competentes empatam). A contribuição é:

1. **Framework unificado:** uma única estrutura algébrica compõe grafos, vetores, tempo e prioridade sem glue code
2. **Travessia híbrida por construção:** hops alternam arestas e pontes de similaridade sem configuração adicional
3. **Entity resolution implícita:** composition + quotient funde entidades semanticamente equivalentes com IDs diferentes (multi-agente)
4. **Associatividade garantida por construção:** ordem de composição é irrelevante (propriedade de design, não teorema)

---

## Testes Unitários

```
30/30 testes passando (pytest tests/ -q)
```

## Limitações Reconhecidas

- Experimentos com vetores sintéticos (32 dims); validação com embeddings reais na Fase 4
- Baselines implementados in-house (não Qdrant/Neo4j reais) — mas com capacidades equivalentes
- Cenários de escala pequena (≤10 nós por memória); escala validada na Fase 2D
