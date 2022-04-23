from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import json
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

@Language.factory("language_detector")
def create_language_detector(nlp, name):
    return LanguageDetector(language_detection_function=None)

en_model_name = 'en_core_web_trf'
print('Loading model {0}...'.format(en_model_name))
en_nlp = spacy.load(en_model_name)

hostName = "localhost"
serverPort = 8080

class NLPServer(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
            
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html><head><title>https://pythonbasics.org</title></head>", "utf-8"))
        self.wfile.write(bytes("<p>Request: %s</p>" % self.path, "utf-8"))
        self.wfile.write(bytes("<body>", "utf-8"))
        self.wfile.write(bytes("<p>This is an example web server.</p>", "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))

    def do_POST(self):
        print(self.path)
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header("Content-type", "application/json")
        self.end_headers()
        if (self.path == '/api/nlp/v1/parse-text'):
            data = {
                'version': '1.0',
                'engine': 'SpaCy',
                'models': (en_model_name),
                'detectedLanguage': 'en',
                'tokens': ('one', 'two', 'three')
            }
            self.wfile.write(bytes(json.dumps(data), "utf-8"))
        else:
            self.wfile.write(bytes(json.dumps({ 'answer': 'text' }), "utf-8"))

if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), NLPServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")