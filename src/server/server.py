import json
from jinja2 import Undefined
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from http.server import BaseHTTPRequestHandler, HTTPServer
from transformers import pipeline
import os
from dotenv import load_dotenv
import requests
import spacy_universal_sentence_encoder
import os, psutil

process = psutil.Process(os.getpid())
print('Used memory in MB: {0}'.format(process.memory_info().rss / 1024 / 1024))

load_dotenv(dotenv_path='../gdmn-nxt/.env')
gdmn_nxt_server = 'http://localhost:{0}'.format(
    os.getenv('GDMN_NXT_SERVER_PORT'))
print('Detected address of gdmn-nxt server: {0}'.format(
    gdmn_nxt_server))

# spacy.prefer_gpu()


@Language.factory('language_detector')
def create_language_detector(nlp, name):
    return LanguageDetector(language_detection_function=None)


en_model_name = 'en_core_web_trf'
en_nlp = Undefined

def load_en_nlp():
    global en_nlp
    if en_nlp is Undefined:
        print('Loading model {0}...'.format(en_model_name))
        en_nlp = spacy.load(en_model_name)
        en_nlp.add_pipe('language_detector')
        print('Used memory in MB: {0}'.format(process.memory_info().rss / 1024 / 1024))
    return en_nlp

ru_model_name = 'ru_core_news_lg'
ru_nlp = Undefined

def load_ru_nlp():
    global ru_nlp
    if ru_nlp is Undefined:
        print('Loading model {0}...'.format(ru_model_name))
        ru_nlp = spacy.load(ru_model_name)
        ru_nlp.add_pipe('language_detector')
        print('Used memory in MB: {0}'.format(process.memory_info().rss / 1024 / 1024))
    return ru_nlp


classifier = Undefined

def load_classifier():
    global classifier
    if classifier is Undefined:
        print('Loading model {0}...'.format("joeddav/xlm-roberta-large-xnli"))
        classifier = pipeline("zero-shot-classification",
                            model="joeddav/xlm-roberta-large-xnli")
        print('Used memory in MB: {0}'.format(process.memory_info().rss / 1024 / 1024))
    return classifier


xx_use_lg = Undefined

def load_xx_use_lg():
    global xx_use_lg
    if xx_use_lg is Undefined:
        print('Loading model xx_use_lg...')
        xx_use_lg = spacy_universal_sentence_encoder.load_model("xx_use_lg")
        print('Used memory in MB: {0}'.format(process.memory_info().rss / 1024 / 1024))
    return xx_use_lg


en_use_lg = Undefined

def load_en_use_lg():
    global en_use_lg
    if en_use_lg is Undefined:
        print('Loading model en_use_lg...')
        en_use_lg = spacy_universal_sentence_encoder.load_model("en_use_lg")
        print('Used memory in MB: {0}'.format(process.memory_info().rss / 1024 / 1024))
    return en_use_lg


hostName = 'localhost'
serverPort = 8080

intent_labels = ["show", "update", "insert", "delete"]

erModel = {}
erModel_fullDbName = ''

class NLPServer(BaseHTTPRequestHandler):
    def getDocAndModelName(self, post_body):
        if post_body['language'] == 'en':
            return load_en_nlp()(post_body['text']), en_model_name
        elif post_body['language'] == 'ru':
            return load_ru_nlp()(post_body['text']), ru_model_name
        else:
            print('Unsupported language: {0}'.format(
                post_body['language']))
            return Undefined, Undefined       

    def checkErModel(self, post_body):
        """Check if ER model is loaded"""
        global erModel
        global erModel_fullDbName

        if erModel_fullDbName != post_body['fullDbName']:
            url = gdmn_nxt_server + '/api/v1/er-model'
            r = requests.get(url) 
            if r.status_code == 200:
                erModel = r.json() 
                erModel_fullDbName = erModel['fullDbName']
                print('Loaded ER model: {0}'.format(erModel_fullDbName))
                print('Entities: {0}...'.format(len(erModel['entities'].keys())))
                print('Domains: {0}...'.format(len(erModel['domains'].keys())))
            else:
                print('Failed to get ER model from {0}'.format(url))
                return False

        if erModel_fullDbName != post_body['fullDbName']:
            print('No erModel for {0}'.format(post_body['fullDbName']))
            return False

        return True

    def do_OPTIONS(self):
        self.send_response(200, 'ok')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(
            bytes('<html><head><title>GDMN NXT Server</title></head>', 'utf-8'))
        self.wfile.write(bytes('<body>', 'utf-8'))
        self.wfile.write(
            bytes('<p>This is an api server for the GDMN NXT platform.</p>', 'utf-8'))
        self.wfile.write(bytes('</body></html>', 'utf-8'))

    def do_POST(self):
        global erModel
        global erModel_fullDbName

        print(self.path)
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        content_len = int(self.headers.get('Content-Length'))
        post_body = json.loads(self.rfile.read(content_len).decode())
        print(post_body)

        if (self.path == '/api/nlp/v1/parse-text'):
            doc, model_name = self.getDocAndModelName(post_body)

            if not doc:
                return

            if not self.checkErModel(post_body):
                return

            sents = []

            for sent in doc.sents:
                tokens = []

                for token in sent:
                    entities = Undefined

                    if token.dep_ == 'obj' and token.pos_ == 'NOUN':
                        """Find an entity in the erModel"""
                        similar = [
                            { 
                                'entity': name, 
                                'lName': entity['lName'],
                                'score': 1.0
                            } 
                            for name, entity in erModel['entities'].items()
                            if 'lName' in entity and token.lemma_ in [s.strip().lower() for s in entity['lName'].split(',')]
                        ]

                        if len(similar) == 0:
                            similarity_model = load_xx_use_lg()
                            token_doc = similarity_model(token.lemma_)
                            similar = [
                                { 
                                    'entity': name, 
                                    'lName': entity['lName'],
                                    'score': similarity_model(entity['lName']).similarity(token_doc) 
                                } 
                                for name, entity in erModel['entities'].items()
                                if 'lName' in entity and similarity_model(entity['lName']).similarity(token_doc) > 0.5
                            ]

                        if len(similar) > 0:
                            similar.sort(key=lambda x: x['score'], reverse=True)
                            entities = similar[0:5]

                    t = {
                        'id': token.i,
                        'token': token.text,
                        'lemma': token.lemma_,
                        'pos': token.pos_,
                        'pos_explain': spacy.explain(token.pos_),
                        'tag': token.tag_,
                        'shape': token.shape_,
                        'is_alpha': token.is_alpha,
                        'is_digit': token.is_digit,
                        'is_currency': token.is_currency,
                        'is_bracket': token.is_bracket,
                        'is_stop': token.is_stop,
                        'dep': token.dep_,
                        'dep_explain': spacy.explain(token.dep_) if token.dep_ != 'ROOT' else '',
                        'morph': token.morph.to_dict(),
                        'start': token._._start,
                        'ent_type': token.ent_type_
                    }

                    if entities is not Undefined:
                        t['entities'] = entities

                    if token.head and token.head.i != token.i:
                        t['head'] = {
                            'id': token.head.i,
                            'ancestors': [an.i for an in token.ancestors],
                            'children': [ch.i for ch in token.children],
                            'conjuncts': [co.i for co in token.conjuncts]
                        }

                    tokens.append(t)

                ents = []

                for ent in sent.ents:
                    e = {
                        'ent': ent.text,
                        'lemma': ent.lemma_,
                        'label': ent.label_,
                        'start_char': ent.start_char,
                        'end_char': ent.end_char
                    }
                    ents.append(e)

                res = load_classifier()(sent.text, intent_labels)

                sents.append({
                    'detectedLanguage': sent._.language,
                    'text': sent.text,
                    'tokens': tokens,
                    'ents': ents,
                    'intent': ({
                        'label': res['labels'][0],
                        'score': res['scores'][0]
                    })
                })

            data = {
                'version': '1.0',
                'engine': 'SpaCy',

                'models': (model_name),
                'text': doc.text,
                'sents': sents
            }
            self.wfile.write(bytes(json.dumps(data), 'utf-8'))
        else:
            self.wfile.write(bytes(json.dumps({'answer': 'text'}), 'utf-8'))


if __name__ == '__main__':
    webServer = HTTPServer((hostName, serverPort), NLPServer)
    print('Server started http://%s:%s' % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print('Server stopped.')
