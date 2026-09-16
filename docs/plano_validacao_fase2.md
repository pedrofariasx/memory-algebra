# Plano de Validação Fase 2: Confiabilidade Empírica

## Objetivo

Elevar a pesquisa de *proof-of-concept interno* para *evidência empírica publicável*, eliminando as cinco limitações identificadas na validação atual.

---

## Estado Atual (Fase 1 — Completo)

| Protocolo | Resultado | Limitação |
|:----------|:----------|:----------|
| 3.1 Associatividade | $d_{max} = 4.4 \times 10^{-17}$ | Conceitos ortogonais, seed único |
| 3.2 Estabilidade | Álgebra 0.0 vs baselines 0.59–1.0 | Baselines não-treinadas, sem significância |
| 3.3 Conflitos | Temporal correto | 2 fatos apenas |
| 3.4 Expressividade | Multi-hop OK | Cadeia linear simples |
| Ground truth Catlab | 10/10 PASS | Valida estrutura, não comportamento real |

---

## Fase 2A: Robustez Estatística

**Objetivo:** Eliminar dependência de seed único e estabelecer significância.

### 2A.1 — Múltiplas Seeds

- Executar todos os protocolos com $S = 50$ seeds distintas.
- Reportar: média, mediana, desvio padrão, IC 95% (bootstrap).
- Critério: todos os protocolos devem passar em $\geq 95\%$ das seeds.

### 2A.2 — Testes de Sensibilidade a Parâmetros

Varredura sistemática:

| Parâmetro | Range | Passos |
|:----------|:------|:-------|
| $\theta$ (limiar de fusão) | $[0.5, 0.99]$ | 10 |
| $\alpha, \beta, \gamma, \delta$ (pesos $d_s$) | simplex | grid 20 |
| $\tau$ (escala temporal) | $[0.1, 10]$ | 8 |
| $\varepsilon$ (poda) | $[0.01, 0.3]$ | 6 |
| dimensão $d$ | $\{8, 16, 32, 64, 128\}$ | 5 |

Critério: associatividade deve ser invariante a todos os parâmetros (propriedade estrutural). Estabilidade pode degradar — mapear a fronteira.

### 2A.3 — Testes Adversariais de Associatividade

- Memórias com vetores idênticos (colisão total).
- Memórias com vetores antiparalelos ($\cos = -1$).
- Memórias com vetores nulos.
- Composição de $N = 1000$ memórias (10× o atual).
- Fusões encadeadas: $A \sim B$, $B \sim C$, mas $A \not\sim C$ (transitividade do limiar).

---

## Fase 2B: Realismo dos Embeddings

**Objetivo:** Sair do regime ortogonal sintético e testar com distribuições realistas.

### 2B.1 — Conceitos Correlacionados

Substituir `orthogonal_concepts()` por geradores com correlação controlada:

```python
def correlated_concepts(n, dim, rho, rng):
    """Gera n vetores com correlação média rho entre pares."""
    base = rng.standard_normal(dim)
    base /= np.linalg.norm(base)
    concepts = []
    for _ in range(n):
        v = rho * base + np.sqrt(1 - rho**2) * rng.standard_normal(dim)
        concepts.append(v / np.linalg.norm(v))
    return concepts
```

Varredura: $\rho \in \{0.0, 0.2, 0.4, 0.6, 0.8, 0.95\}$.

**Hipótese de falha:** com $\rho > 0.7$, o limiar $\theta = 0.85$ pode fundir conceitos distintos, destruindo a recuperabilidade. Se confirmado, a álgebra precisa de $\theta$ adaptativo ou fusão por componente.

### 2B.2 — Embeddings Reais (Sentence-Transformers)

- Usar `sentence-transformers/all-MiniLM-L6-v2` (dim=384) ou `paraphrase-multilingual-MiniLM-L12-v2`.
- Gerar memórias a partir de frases reais em português/inglês.
- Incluir: sinônimos (alta similaridade), antônimos, frases ambíguas.
- Testar se $\theta$ calibrado para embeddings reais preserva a separabilidade.

### 2B.3 — Ruído e Degradação

- Adicionar ruído gaussiano com $\sigma \in \{0.01, 0.05, 0.1, 0.2, 0.5\}$.
- Testar dropout de arestas: remover $p \in \{0.1, 0.3, 0.5\}$ das arestas aleatoriamente.
- Testar corrupção temporal: embaralhar $T$ em $k\%$ dos nós.

---

## Fase 2C: Baselines Competitivas

**Objetivo:** Substituir baselines triviais por competidores reais.

### 2C.1 — Baselines a Implementar

| Baseline | Descrição | Esforço |
|:---------|:----------|:--------|
| LSTM treinada | Treinar LSTM para memorizar sequência de vetores, medir recall | Médio |
| DNC (Differentiable Neural Computer) | Memória externa com endereçamento por conteúdo | Alto |
| Memória Transformer | Cross-attention sobre buffer de memórias (sem posição fixa) | Médio |
| Retrieval-augmented | Buffer + kNN retrieval (FAISS) — baseline não-paramétrico | Baixo |
| EWC (Elastic Weight Consolidation) | Regularização anti-esquecimento em rede simples | Médio |

### 2C.2 — Protocolo de Comparação Justa

Para cada baseline:
1. Mesmo corpus de memórias (mesmos vetores, mesma ordem).
2. Mesmas consultas $q_i$.
3. Mesma métrica $d_s$.
4. Mesmo orçamento computacional (FLOPs ou wall-clock).
5. Reportar: curva de decaimento, tempo de inserção, tempo de retrieval, memória ocupada.

### 2C.3 — Treinamento da LSTM Baseline

A LSTM atual usa pesos aleatórios. Para comparação justa:
- Treinar com objetivo: dado $x_t$, reconstruir $x_1$ (primeira memória).
- Arquitetura: hidden=dim, 1 camada (equivalente ao teste atual).
- Otimizador: Adam, lr=1e-3, 500 steps por sequência.
- Isso dá à LSTM a *melhor chance* de reter informação — se ainda perder, o argumento é mais forte.

---

## Fase 2D: Escala

**Objetivo:** Validar que a álgebra não degrada com $N$ grande.

### 2D.1 — Complexidade Empírica

Medir tempo de `compose` e `retrieve` para:

| $N$ (memórias) | Nós totais | Tempo compose | Tempo retrieve |
|:---------------|:-----------|:--------------|:---------------|
| 100 | ~300 | ? | ? |
| 1.000 | ~3.000 | ? | ? |
| 10.000 | ~30.000 | ? | ? |
| 100.000 | ~300.000 | ? | ? |

### 2D.2 — Otimizações Necessárias

- `quotient()` é $O(N^2)$ — substituir por LSH (Locality-Sensitive Hashing) para candidatos.
- `retrieve()` recomputa o quociente a cada consulta — cache incremental.
- Para $N > 10^4$: portar para Rust com adjacência esparsa (já planejado no `instructions.md`).

### 2D.3 — Critério de Aceitação

- `compose`: $O(1)$ amortizado (já é, por construção).
- `retrieve`: $O(N \log N)$ ou melhor com LSH.
- Associatividade: invariante a $N$ (propriedade algébrica, não deve degradar).

---

## Fase 2E: Avaliação Extrínseca

**Objetivo:** Demonstrar utilidade em tarefa real, não apenas propriedades internas.

### 2E.1 — Tarefa: QA Factual com Memória Composicional

Construir benchmark sintético com estrutura de conhecimento:

```
Fatos: "Brasília é capital do Brasil" (t=1)
       "Lula é presidente do Brasil" (t=2)
       "Brasil faz fronteira com Argentina" (t=3)
       ...
Consulta: "Qual a capital do país onde Lula é presidente?"
```

- Multi-hop: requer composição de 2+ fatos.
- Temporal: incluir fatos que mudam ("presidente em 2022 vs 2024").
- Métrica: accuracy exata + $d_s$ parcial.

### 2E.2 — Tarefa: Diálogo Multi-turno com Consistência

- Simular 20 turnos de diálogo com fatos inseridos progressivamente.
- Verificar consistência: o sistema contradiz fatos anteriores?
- Comparar: álgebra vs. contexto deslizante (últimos $k$ turnos).

### 2E.3 — Tarefa: Atualização de Conhecimento

- Inserir fato, depois inserir correção.
- Consultar antes e depois da correção.
- Verificar: resposta muda corretamente? Histórico acessível?

---

## Cronograma Estimado

| Fase | Tarefas | Duração | Dependência |
|:-----|:--------|:--------|:------------|
| 2A | Seeds, sensibilidade, adversariais | 1 semana | Nenhuma |
| 2B | Conceitos correlacionados, embeddings reais, ruído | 2 semanas | Nenhuma |
| 2C | Baselines (retrieval-augmented primeiro, LSTM treinada, Transformer) | 2-3 semanas | Nenhuma |
| 2D | Escala + LSH + benchmark de tempo | 1 semana | Nenhuma |
| 2E | QA sintético + diálogo + atualização | 2-3 semanas | 2B, 2C |
| **Total** | | **6-8 semanas** | |

---

## Critérios de Publicação

A pesquisa estará pronta para submissão quando:

1. [ ] Associatividade confirmada em 50/50 seeds com conceitos correlacionados ($\rho = 0.8$).
2. [ ] Estabilidade superior a TODAS as baselines com $p < 0.01$ (teste de Wilcoxon).
3. [ ] Recuperabilidade mantida com embeddings reais (sentence-transformers).
4. [ ] Escala até $N = 10^4$ com retrieve < 1s.
5. [ ] QA factual multi-hop: accuracy > baseline retrieval-augmented.
6. [ ] Nenhuma falha catastrófica nos testes adversariais.

---

## Riscos e Mitigações

| Risco | Impacto | Mitigação |
|:------|:--------|:----------|
| $\theta$ fixo falha com embeddings correlacionados | Alto | $\theta$ adaptativo por cluster (percentil local) |
| Retrieval $O(N^2)$ não escala | Médio | LSH + cache incremental |
| Baselines treinadas superam a álgebra | Alto (invalida claim) | Publicar como resultado negativo + analisar por quê |
| QA sintético muito fácil/difícil | Médio | Calibrar dificuldade com baseline de oracle |
| Contradição com ground truth Catlab em casos edge | Alto | Corrigir álgebra antes de prosseguir |

---

## Entregáveis por Fase

- **2A:** `experiments/phase2a_seeds.py` + `results_phase2a.json`
- **2B:** `experiments/phase2b_embeddings.py` + `results_phase2b.json`
- **2C:** `memory_algebra/baselines_extended.py` + `experiments/phase2c_comparison.py`
- **2D:** `experiments/phase2d_scale.py` + gráfico tempo vs N
- **2E:** `experiments/phase2e_qa.py` + dataset sintético em `data/`
- **Relatório:** `docs/relatorio_fase2.md` com todos os resultados e análise
