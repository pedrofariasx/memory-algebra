[English version](README.en.md)

# memory-algebra

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/pedrofariasx/memory-algebra/actions/workflows/tests.yml/badge.svg)](https://github.com/pedrofariasx/memory-algebra/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-30%20passed-brightgreen)](tests/)
[![Preprint](https://img.shields.io/badge/preprint-CC--BY--4.0-orange)](https://zenodo.org/records/22815162)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22815162.svg)](https://doi.org/10.5281/zenodo.22815162)

Uma **álgebra de memórias** para sistemas de IA que resolve o problema de **interferência catastrófica** (catastrophic forgetting): composições sucessivas de memórias sem destruição das representações anteriores. Implementação em Python com provas formais de associatividade exata, validação empírica contra baselines (LSTM, soma vetorial, kNN), e preprint disponível em PT-BR e EN.

## Hipótese Central

Sistemas de memória para IA atuais (RNNs, buffers FIFO, soma de vetores) sofrem de **interferência catastrófica**: composições sucessivas diluem representações anteriores. Este projeto propõe uma **álgebra de memórias** estruturada em quatro componentes $(V, G, T, P)$ que satisfaz simultaneamente:

| Propriedade | Descrição |
|:-----------:|:----------|
| $\boxed{A}$ Associatividade | $(m_1 \oplus m_2) \oplus m_3 \cong m_1 \oplus (m_2 \oplus m_3)$ |
| $\boxed{R}$ Recuperabilidade | $d_s(m, \rho(M, q)) \to 0$ para qualquer memória $m$ inserida |
| $\boxed{S}$ Estabilidade | $\|M_t\| < C$ sob poda $\Pi_C$ |
| $\boxed{P}$ Preservação | $\otimes$ não destrói a estrutura recuperável |
| $\boxed{E}$ Eficiência | Operações tratáveis sobre representações discretas |

---

## Estrutura dos Componentes

Uma memória é a tupla $m = (N, v, E, \tau, w)$:

- **$V$** (`vectors`): embeddings densos em $\mathbb{R}^d$ — conteúdo semântico.
- **$G$** (`edges`): grafo rotulado entre nós — relações estruturais.
- **$T$** (`times`): timestamps — resolução de conflitos temporais.
- **$P$** (`weights`): utilidade em $[0,1]$ — controle de retenção/esquecimento.

---

## Operações Algébricas

| Operação | Símbolo | Implementação | Propriedade |
|:---------|:-------:|:--------------|:------------|
| Compose | $\oplus$ | União disjunta atômica + quociente por fusão $\theta$ (`compose`, `quotient`) | Associatividade exata |
| Retract | $\ominus$ | Remoção de nós + projeção ortogonal (`retract`) | Consistência via lentes |
| Transform | $\otimes$ | Matriz $W$ sobre vetores + renomeação de arestas (`transform`) | Functorialidade |
| Retrieve | $\rho$ | Score cosseno × fator temporal × peso + BFS no quociente (`retrieve`) | Recuperabilidade |
| Prune | $\Pi_C$ | Esquecimento por $w < \varepsilon$ + atalhos + teto de capacidade (`prune`) | Estabilidade |

---

## Resultados Empíricos

### 3.1 — Associatividade (N=100 memórias)

Três caminhos de composição (esquerda, direita, paralelo) produzem **assinaturas idênticas**.

```
max_distancia(esquerda, direita)  = 4.44e-17
max_distancia(esquerda, paralelo) = 4.44e-17
```

### 3.2 — Interferência Catastrófica (50 composições)

Capacidade de recuperar a primeira memória ($m_1$) após 50 composições subsequentes:

| Método | $d_s$ final | |
|:-------|:-----------:|:--|
| **Álgebra $(V,G,T,P)$** | **0.0** | recuperação perfeita |
| Soma vetorial | 0.929 | diluição total |
| LSTM (hidden=dim) | 0.588 | esquecimento parcial |
| Buffer FIFO (cap=10) | 1.0 | perda total |

### 3.3 — Resolução de Conflitos

```
M = compose("João é médico", t=1, "João virou engenheiro", t=2)
```

- Consulta em $t=2$ → retorna "engenheiro" primeiro ✓
- Consulta em $t=1$ → retorna "médico" primeiro ✓
- Histórico preservado (ambos os fatos existem em $M$) ✓
- Entidade "João" fundida no quociente ✓

### 3.4 — Expressividade Multi-hop

```
m1(A→B), m2(B→C), m3(C→D) ⇒ consulta "A?" com hops=3 recupera D
```

Caminho transitivo inferido sem aresta direta A→D ✓

### Leis das Lentes

- **Put-Get:** $d_s(\Delta, get(put(M,q,\Delta), q_\Delta)) = 0.0$ ✓
- **Get-Put:** $put(M, q, get(M,q)) = M$ (vistas isoladas) ✓

---

## Fundamentação Formal

### Teoria (Pilar 1)

- Composição modelada como **pushout** em categoria de grafos tipados.
- Prova de associatividade: união disjunta em $\mathbf{Set}$ + fecho transitivo único de $R_\theta$.
- Lentes delta assimétricas com leis Put-Get e Get-Put.
- Métrica híbrida $d_s = \alpha d_V + \beta d_G + \gamma d_T + \delta d_P$.

Detalhes completos em [`docs/formalizacao.md`](docs/formalizacao.md).

### Ground Truth Categórico (Catlab.jl)

`formal/catlab_ground_truth.jl` valida pushouts exatos com `is_isomorphic`:

```
[PASS] associativity: ((m1⊕m2)⊕m3) ≅ (m1⊕(m2⊕m3))
[PASS] commutativity: (m2⊕m1) ≅ (m1⊕m2)
[PASS] identity: m1 ⊕ ∅ ≅ m1
ALL CHECKS PASSED
```

### Camada Diferenciável (JAX)

`memory_algebra/diff.py` implementa a álgebra como funções puras sobre arrays JAX:
- `compose`: soma de matrizes de adjacência — associatividade matricial exata.
- `retrieve`: softmax cosseno × fator temporal × peso.
- `sample_edges`: Gumbel-Softmax para amostragem diferenciável de arestas.
- Gradientes fluem através de `compose` e `retrieve` (verificado com `jax.grad`).

---

## Estrutura do Repositório

```
memory-algebra/
├── memory_algebra/                  # Package Python (álgebra)
│   ├── core.py             # MemoryObject, compose, quotient, retrieve, retract, transform
│   ├── diff.py             # Álgebra diferenciável (JAX)
│   ├── metric.py           # d_s com pesos contextuais
│   ├── lens.py             # Lentes get/put
│   ├── stability.py        # Poda Π_C
│   ├── baselines.py        # LSTM baseline
│   └── builder.py          # Construtores e utilidades
├── tests/                  # 30 testes (pytest + hypothesis)
├── experiments/
│   ├── run_validation.py   # Protocolos 3.1–3.4 + lentes
│   └── results.json        # Resultados numéricos
├── formal/
│   └── catlab_ground_truth.jl  # Validação categórica (Catlab.jl)
├── docs/
│   └── formalizacao.md     # Provas e correspondência teoria↔código
├── instructions.md         # Plano de execução original
└── pyproject.toml
```

---

## Execução

```bash
# Testes (30 testes)
.venv/bin/python -m pytest

# Validação empírica completa
.venv/bin/python experiments/run_validation.py

# Ground truth categórico (requer Julia + Catlab 0.15)
julia formal/catlab_ground_truth.jl
```

---

## Dependências

| Pacote | Uso |
|:-------|:----|
| numpy | Vetores, álgebra linear |
| networkx | Análise estrutural de grafos |
| jax | Camada diferenciável |
| pytest + hypothesis | Testes de propriedade |
| Catlab.jl 0.15 (Julia) | Ground truth categórico |

---

## Conclusão

A álgebra $(V, G, T, P)$ resolve o trade-off central: **composição associativa sem destruição da representação**. Os resultados demonstram que a estrutura de grafo + tempo + utilidade protege vetores individuais da diluição — propriedade que nenhuma baseline testada (soma vetorial, LSTM, FIFO) consegue manter. A camada diferenciável abre caminho para integração com aprendizado neural, mantendo as garantias algébricas.

---

## Como Citar

Farias, P. F. R. (2026). *An Algebra of Memory: Formal Foundations and Empirical Validation of Associative Composition Without Catastrophic Interference*. Zenodo. [https://doi.org/10.5281/zenodo.22815162](https://doi.org/10.5281/zenodo.22815162)
