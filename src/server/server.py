import json
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from http.server import BaseHTTPRequestHandler, HTTPServer

@Language.factory('language_detector')
def create_language_detector(nlp, name):
    return LanguageDetector(language_detection_function=None)

alphabets = {
    'ru': 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
    'en': 'abcdefghijklmnopqrstuvwxyz',
    'by': 'абвгдеёжзійклмнопрстуўфхцчш''ыьэюя'
}    

en_model_name = 'en_core_web_trf'
print('Loading model {0}...'.format(en_model_name))
en_nlp = spacy.load(en_model_name)
en_nlp.add_pipe('language_detector')

ru_model_name = 'ru_core_news_lg'
ru_nlp = spacy.load(ru_model_name)
print('Loading model {0}...'.format(ru_model_name))
ru_nlp.add_pipe('language_detector')


hostName = 'localhost'
serverPort = 8080

class NLPServer(BaseHTTPRequestHandler):
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
        self.wfile.write(bytes('<html><head><title>https://pythonbasics.org</title></head>', 'utf-8'))
        self.wfile.write(bytes('<p>Request: %s</p>' % self.path, 'utf-8'))
        self.wfile.write(bytes('<body>', 'utf-8'))
        self.wfile.write(bytes('<p>This is an example web server.</p>', 'utf-8'))
        self.wfile.write(bytes('</body></html>', 'utf-8'))

    def do_POST(self):
        print(self.path)
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        if (self.path == '/api/nlp/v1/parse-text'):
            content_len = int(self.headers.get('Content-Length'))
            post_body = json.loads(self.rfile.read(content_len).decode())
            print(post_body)

            if post_body['language'] == 'en': 
              doc = en_nlp(post_body['text'])
              model = en_model_name
            elif post_body['language'] == 'ru':   
              doc = ru_nlp(post_body['text'])
              model = ru_model_name
            else:
              print('Unsupported language: {0}'.format(post_body['language']))    
              return

            sents = []

            for sent in doc.sents:
                tokens = []

                for token in sent:
                    t = {
                        'id': token.i,
                        'token': token.text,
                        'lemma': token.lemma_,
                        'pos': token.pos_,
                        'tag': token.tag_,
                        'shape': token.shape_,
                        'is_alpha': token.is_alpha,
                        'is_stop': token.is_stop,
                        'dep': token.dep_,
                        'morph': token.morph.to_dict()
                    }

                    if token.head:
                        t['head'] = {
                            'id': token.head.i,
                            'ancestors': [an.i for an in token.ancestors],
                            'children': [ch.i for ch in token.children],
                            'conjuncts': [co.i for co in token.conjuncts]
                        }

                    tokens.append(t)

                sents.append({
                    'detectedLanguage': sent._.language,
                    'text': sent.text,
                    'tokens': tokens
                })

            data = {
                'version': '1.0',
                'engine': 'SpaCy',

                'models': (model),
                'text': doc.text,
                'sents': sents
            }
            self.wfile.write(bytes(json.dumps(data), 'utf-8'))
        else:
            self.wfile.write(bytes(json.dumps({ 'answer': 'text' }), 'utf-8'))

if __name__ == '__main__':        
    webServer = HTTPServer((hostName, serverPort), NLPServer)
    print('Server started http://%s:%s' % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print('Server stopped.')