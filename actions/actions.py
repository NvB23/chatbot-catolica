from typing import Any, Text, Dict, List
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk import Action, Tracker
from sentence_transformers import SentenceTransformer
import faiss, json


modelo = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index('docs.index')

with open('docs.json') as doc:
    documentos = json.load(doc)

class ActionBuscaDocumento(Action):

    def name(self) -> Text:
        return "action_buscar_documento"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        pergunta_usuario = tracker.latest_message['text']

        vetor_pergunta_usuario = modelo.encode([pergunta_usuario])

        _, posicoes = index.search(vetor_pergunta_usuario, 1)

        indice = posicoes[0][0]
        resposta = documentos[indice]['conteudo']

        dispatcher.utter_message(text=resposta)

        return []




# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
