from sentence_transformers import SentenceTransformer
import faiss
import json
from datetime import datetime

documentos = [
    {"titulo": "Edital 01/2025", "conteudo": "A matrícula ocorrerá entre 10 e 15 de março."},
    {"titulo": "Calendário Acadêmico", "conteudo": "O semestre inicia em 01 de agosto de 2025."},
    {"titulo": "Edital Vestibular 2025.2", "conteudo": "O período de inscrisão ocorrerá entre os meses junho e julho. Nesse período, poderá ser realizado a prova para os Cursos de Direito e Ciência da Computação."},
    {"conteudo": "O prazo para trancamento de matrícula vai até 30 de setembro."},
    {"conteudo": "As provas finais ocorrerão entre os dias 10 e 20 de dezembro."},
    {"conteudo": "O calendário acadêmico está disponível no portal do aluno."},
    {"conteudo": "As inscrições para o processo seletivo encerram no dia 15 de julho."},
    {"conteudo": "Os editais de extensão foram atualizados em 25 de maio."},
    {"conteudo": "O resultado das bolsas será divulgado no dia 10 de junho."},
    {"conteudo": "O período de rematrícula vai de 1 a 10 de julho."},
    {"conteudo": "A colação de grau dos formandos de 2025.1 será no dia 22 de julho."},
    {"conteudo": "As aulas práticas dos cursos de saúde retornam em 12 de agosto."},
    {"conteudo": "A semana acadêmica acontecerá entre os dias 20 e 24 de outubro."},
    {"conteudo": "O prazo para submissão de trabalhos científicos é 15 de setembro."},
    {"conteudo": "O resultado final do ENADE será publicado no site do MEC."},
    {"conteudo": "A biblioteca funcionará em horário especial durante as férias."},
    {"conteudo": "O novo regulamento de TCC está disponível na área do aluno."}
]

modelo = SentenceTransformer("all-MiniLM-L6-v2")
textos = [doc['conteudo'] for doc in documentos]
vetores = modelo.encode(textos)

index = faiss.IndexFlatL2(vetores.shape[1])
index.add(vetores)
print(datetime.now())
faiss.write_index(index, 'docs.index')
with open('docs.json', 'w') as doc:
    json.dump(documentos, doc)