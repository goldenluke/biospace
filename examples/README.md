# Exemplos

Scripts standalone, comentados, testados — cada um roda sozinho e
imprime resultado no terminal. Complementam (não substituem) a suíte de
testes em `tests/`: aqui o objetivo é ser lido e explorado por humanos,
lá é travar comportamento como regressão.

```bash
python3 examples/01_new_disease_from_scratch.py
python3 examples/02_causal_inference_pipeline.py
python3 examples/03_dynamics_and_stability.py
python3 examples/04_geometry_comparison.py
python3 examples/05_diabetes_toy_disease.py
python3 examples/06_representation_learning.py
python3 examples/07_knowledge_graph.py
python3 examples/08_gnn.py
python3 examples/09_curvature_and_metastability.py
python3 examples/10_digital_twin_simulation.py
python3 examples/11_masked_feature_prediction.py
```

| Script | O que mostra |
|---|---|
| `01_new_disease_from_scratch.py` | O núcleo é genérico de verdade — constrói uma doença nova (glicemia) sem tocar no plugin sleep |
| `02_causal_inference_pipeline.py` | Balanceamento → escore de propensão → pareamento → efeito, validado contra um efeito causal verdadeiro conhecido (só possível em dado sintético) |
| `03_dynamics_and_stability.py` | `EvolutionOperator` aprende a dinâmica espontânea da coorte e recupera corretamente o parâmetro usado para gerar os dados |
| `04_geometry_comparison.py` | A mesma dupla de pontos sob 4 geometrias diferentes — nenhuma é "a" distância certa |
| `05_diabetes_toy_disease.py` | Segundo plugin de doença COMPLETO (6 domínios, latente, gerador longitudinal) — todo o ferramental (contratos, fenotipagem, causal) funciona sem alteração no núcleo |
| `06_representation_learning.py` | Autoencoder não linear vs. PCA sobre o RepresentationSpace — vence em estrutura latente não linear conhecida, mas PERDE nos dados reais de SAOS (achado real: poucos dados para o não linear valer a pena) |
| `07_knowledge_graph.py` | O paciente deixa de ser vetor e passa a ser rede — grafo interno (domínios/Features/correlações reais) + grafo de similaridade populacional (a estrutura que uma GNN futura consumiria) |
| `08_gnn.py` | GCN (Kipf & Welling) em NumPy puro — padrão de cruzamento: com muitos rótulos o grafo atrapalha, com poucos rótulos ajuda muito |
| `09_curvature_and_metastability.py` | As 3 formas independentes de curvatura do projeto (temporal, densidade populacional, estrutural via grafo) e metaestabilidade via poços de potencial |
| `10_digital_twin_simulation.py` | Gêmeo digital com simulação em conjunto (múltiplos futuros, incerteza real) — validado contra a variância estacionária teórica conhecida de um processo de Ornstein-Uhlenbeck |
| `11_masked_feature_prediction.py` | Protótipo de arquitetura de "foundation model" (masked feature prediction, estilo BERT) — aprende estrutura real onde existe, não inventa onde não existe; roda sem alteração em dois plugins de doença diferentes |

Para exemplos com o plugin de sono especificamente (dados reais e
sintéticos, pipeline completo), veja `demo_sleep.py` e
`run_real_cohort.py` na raiz do repositório.

Nenhum destes scripts depende de dados reais de paciente — todos
constroem coortes sintéticas pequenas e rápidas (segundos, não minutos).
