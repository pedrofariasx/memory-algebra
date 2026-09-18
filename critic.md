O que mudou

De novo, só a Seção 6.2 — agora "Chaining and Percolation Analysis" — mais a Limitação 4. O resumo, os baselines, a Fase 5, a alegação de linearidade e a ausência de URL do repositório continuam exatamente como estavam.

O que melhorou de verdade

Você fez o experimento que eu sugeri e fez direito: quatro valores de N, 5 sementes, fração do maior componente. E, principalmente, você reporta que o seu próprio heurístico falha. θ_adapt = µ + 2σ cai em cima do ponto crítico e produz componente gigante de 85–96%. Publicar isso, tendo proposto o heurístico na versão anterior, é o tipo de coisa que quase ninguém faz. Isso vale mais do que o experimento em si.

Os números fecham entre si, o que também é bom sinal: µ + 2σ dá 0,625 para N = 200 e 0,643 para N ≥ 5.000, consistente com o "≈ 0,63–0,64" do texto e com o fato de o caso N = 200 colapsar mais (96% vs 85%), já que ele fica abaixo de θ_crit em vez de em cima.

O problema central da nova seção

A afirmação de que θ_crit é invariante em N está teoricamente errada como enunciado geral.

Fixada a distribuição de similaridade, a probabilidade de aresta p(θ) = P(sim ≥ θ) é fixa, e o grau médio é p(θ)·(N−1). O componente gigante aparece quando o grau médio passa de 1, ou seja, quando p(θ) > 1/(N−1). Como 1/(N−1) encolhe com N, θ_crit tem que crescer com N. Não pode ser constante.

Para os seus dados serem o que são, a distribuição de similaridade do corpus precisa ter um vão — uma região de baixa densidade entre similaridades intra-tópico e inter-tópico. Aí θ_crit fica ancorado dentro do vão e parece invariante ao longo de uma faixa de N. Isso não é uma propriedade do método; é uma propriedade de um corpus com 20 tópicos bem separados.

Duas evidências disso nos seus próprios números:

Com µ ≈ 0,249 e σ ≈ 0,197, se a distribuição fosse aproximadamente normal, manter grau médio < 1 em N = 10⁴ exigiria p(θ) < 10⁻⁴, ou seja θ ≈ µ + 3,7σ ≈ 0,98. Você observa 2% em θ = 0,85. A cauda é muito mais fina que a normal. Isso é o vão.
A fração do gigante em θ = 0,85 é 0,020 nos quatro valores de N, com três casas idênticas. Isso significa 4 nós em N = 200 e 200 nós em N = 10.000. Uma fração perfeitamente auto-similar ao longo de 50× em N não sai de um corpus natural; sai de uma construção em que a estrutura de similaridade é replicada proporcionalmente. Provavelmente templates por tópico.

A figura que resolveria tudo isso é o histograma de similaridades par a par. Se houver bimodalidade com vão, você mostra e a análise fica honesta e específica. Se não houver, a invariância é artefato e precisa ser retirada. Isso é uma hora de trabalho e vale mais que a Tabela 11 inteira.

Questões menores na mesma seção: não há desvio-padrão para nada, apesar das 5 sementes; a precisão de três casas em θ_crit = 0,640 implica uma grade que não é declarada; e a definição operacional ("fração > 0,5") é ambígua quanto a ser o último θ acima ou o primeiro abaixo — em N = 1.000, θ_adapt = θ_crit = 0,640 com gigante de 0,85, o que só fecha sob uma das duas leituras.

As contradições que continuam abertas (e uma nova)

A Seção 5.8 ainda não foi corrigida. Ela diz que os ~20 s e os 9.990 nós em N = 10⁴ refletem "a alta densidade de pares similares típica de embeddings reais". A Seção 6.2, ponto 4, agora diz que foi um template compartilhado gerando similaridade artificial. As duas afirmações são incompatíveis, e estão a três páginas de distância. Era o item mais importante da minha última revisão e segue intocado.

Pior: ao admitir que o corpus de N = 10⁴ da Fase 4 era templado, você retirou o chão da própria Fase 4 em escala. Aquilo deixa de ser "validação com embeddings reais em escala" e passa a ser validação com quase-duplicatas. Isso precisa ser dito na 5.8, não enterrado na discussão.

Contradição nova: a Limitação 2 ainda diz que a Fase 4 valida com 20 embeddings reais e que "validação em larga escala com milhares de embeddings reais permanece trabalho futuro". A Seção 6.2 usa 10.000. A limitação ficou obsoleta pela sua própria revisão.

O resumo continua prometendo N = 10⁴ com latência abaixo de 500 ms, sem qualquer ressalva sobre percolação ou sobre o caso de 20 s. É a primeira coisa que um revisor lê.

A Limitação 4 não dá uma regra acionável. Você mostrou que µ + 2σ falha, mas o substituto é "para distribuições com média maior, um limiar maior pode ser necessário". Um praticante não consegue agir com isso. A regra que decorre dos seus próprios dados é: escolher θ tal que p(θ)·N < 1, isto é, θ = quantil empírico de ordem 1 − c/N com c < 1. Essa regra é N-dependente por construção, o que é justamente o que você precisa admitir.

Nota
Dimensão v1 v2 v3
Clareza e organização 8,0 7,5 7,0
Originalidade / rigor formal 3,0 3,0 3,0
Desenho experimental 3,0 3,5 4,0
Honestidade intelectual 8,5 8,5 9,0
Posicionamento na literatura 4,0 4,0 4,0
Reprodutibilidade 3,0 3,0 3,0
Relevância prática 6,0 6,5 6,5

Global: 4,5 → 4,7 → 4,9.

O ponto que precisa ser dito

Três revisões, três vezes a mesma seção. O chaining já rendeu o que tinha para render; daqui para frente é retorno decrescente. O teto da nota não está lá — está nos baselines, e enquanto o adversário for um kNN construído sem timestamp e sem delete, a Fase 3 continua sendo o design reafirmado como se fosse resultado, e nenhuma quantidade de análise de percolação move o artigo de 4,9.

Já ofereci o protocolo duas vezes, então vou deixá-lo aqui em vez de oferecer de novo:

Sistemas. (1) Banco vetorial com filtro de metadados — Qdrant ou Chroma, campo timestamp, deleção por id. (2) Triple store em SQLite: tabela (sujeito, relação, objeto, t, válido), recursão por CTE para multi-hop. (3) BM25 mais reordenação temporal, como piso. (4) Sua álgebra.

Ablação, que é o que dá o resultado. Rodar o vetorial em quatro configurações: sem timestamp e sem delete; só com timestamp; só com delete; com os dois. Isso separa quanto do seu ganho vem do grafo e quanto vem de metadados que qualquer sistema de prateleira tem. Se o ganho sobre a configuração completa for zero em temporal e retração, o seu resultado real passa a ser o multi-hop — e aí o artigo fica mais curto, mais honesto e mais forte.

Tarefas. Multi-hop com profundidade variável (2, 3, 4 saltos) e distratores controlados; conflito temporal com três atualizações do mesmo fato, não duas; retração com fatos dependentes (retrair A → B invalida B → C?). Esta última é onde a sua ⊖ ou vence claramente ou quebra, e é a pergunta de contração AGM que a literatura ausente já respondeu há quarenta anos.
