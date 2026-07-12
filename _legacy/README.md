# _legacy/

Arquivos órfãos encontrados numa auditoria geral do projeto (12 de
julho de 2026) — todos investigados individualmente antes de mover
(diff de conteúdo, não só nome de arquivo). Nenhum tem conteúdo que
não esteja já superado, incorporado ou disponível de forma mais
completa em algum lugar do projeto organizado. Movidos aqui, não
apagados, para preservar histórico.

- **`meta_modelo_biologico/`** — o protótipo original do meta-modelo
  (`metamodel.py`, single-file, só 4 contratos — hoje são 11 em
  `biospace/core/`; `demo_saos.py`, dado sintético). Claramente
  anterior à organização em `biospace/`. Nada referencia este
  diretório em nenhum lugar do projeto atual.
- **`artigo_diabetes_completo.md`** — uma síntese mais ampla que o
  artigo "oficial" (`artigo_diabetes/artigo_diabetes.md`), mas escrita
  **antes** da correção do bug de sensibilidade (ainda cita 66,6%, o
  valor correto é 75,0%) e antes de lipídios/ICD-9/equidade —
  desatualizada, não uma versão mais nova.
- **`artigo_saos_biospace.md`** — o rascunho mais antigo de todos os
  artigos sobre SAOS (6 de julho, 3 dias antes de qualquer outro em
  `artigo/`), estrutura rasa (7 seções básicas). Todo o conteúdo está
  coberto, e muito expandido, em `artigo/artigo_saos_sem_coorte.md`.
- **`linkedin_post_3_bug_temporalidade.md`** — versão curta (26
  linhas) da história do bug `as_of`/Contrato de Temporalidade.
  `artigo/post_linkedin_temporalidade.md` é a mesma história, 152
  linhas, muito mais desenvolvida.
- **`dados_anonimizados_70.xlsx`** — **dado sintético/fabricado**
  (IDs no padrão `cohort70_001`, `cohort70_002`...), não paciente
  real. Nome de arquivo enganoso (diz "_70" mas tem 355 registros
  fabricados) — provavelmente um testbed de desenvolvimento anterior
  ao upload dos arquivos Excel reais de SAOS. Confirmado sem risco de
  privacidade: não é dado clínico de verdade.
