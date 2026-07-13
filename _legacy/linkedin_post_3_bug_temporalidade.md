Achei um bug real testando uma das propriedades formais que eu mesmo tinha definido. E ele me ensinou algo sobre o que "trajetória clínica" realmente significa.

A teoria por trás do projeto exige que cada nova observação produza uma atualização da trajetória do paciente — nunca contaminada por informação futura. Parece óbvio escrito assim. Mas "óbvio" e "garantido pelo código" são coisas diferentes, então escrevi um verificador para provar isso formalmente: alimentar o sistema com observações fora de ordem cronológica e checar se ele continuava correto.

O resultado foi o oposto do que eu esperava.

O sistema "quebrado" — aquele que não ordenava nada antes de processar — passava no teste.

O sistema "correto" — que eu tinha certeza que estava certo — falhava.

Causa raiz: o `timestamp` de cada observação era usado só como rótulo do ponto na trajetória, nunca como um corte real sobre quais observações entravam no cálculo. Quando uma observação antiga chegava ao sistema depois de uma mais recente já processada (o que acontece o tempo todo em dado clínico real — exames são digitados fora de ordem, resultados de laboratório atrasam), o sistema silenciosamente "espiava o futuro" ao recalcular o passado.

Correção: um parâmetro `as_of`, propagado por toda a cadeia de transformação, filtrando exatamente quais observações eram conhecidas até aquele instante específico — não todas as observações que o sistema já tinha visto até agora.

```python
def transform(self, system, timestamp=None):
    as_of = timestamp or datetime.now()
    known = [obs for obs in system.observations if obs.timestamp <= as_of]
    ...
```

Retrocompatibilidade total em toda a base de pacientes já testada até então — a correção só muda o comportamento exatamente no caso que estava genuinamente quebrado (dado chegando fora de ordem). Em todo o resto, o resultado é idêntico.

Isso é o tipo de bug que só aparece quando você trata propriedade teórica como contrato verificável — algo que um teste automatizado pode provar ou refutar — e não como intenção de design documentada num README que ninguém confere depois.

Trabalham com dados longitudinais? Curioso se "espiar o futuro" é um problema que já morderam vocês também.
