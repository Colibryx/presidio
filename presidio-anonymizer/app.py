"""REST API server for anonymizer."""

import json
import logging
import os
from logging.config import fileConfig
from pathlib import Path

from flask import Flask, Response, jsonify, request
from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine
from presidio_anonymizer.entities import InvalidParamError
from presidio_anonymizer.services.app_entities_convertor import AppEntitiesConvertor
from werkzeug.exceptions import BadRequest, HTTPException

DEFAULT_PORT = "3000"

LOGGING_CONF_FILE = "logging.ini"

WELCOME_MESSAGE = r"""
 _______  _______  _______  _______ _________ ______  _________ _______
(  ____ )(  ____ )(  ____ \(  ____ \\__   __/(  __  \ \__   __/(  ___  )
| (    )|| (    )|| (    \/| (    \/   ) (   | (  \  )   ) (   | (   ) |
| (____)|| (____)|| (__    | (_____    | |   | |   ) |   | |   | |   | |
|  _____)|     __)|  __)   (_____  )   | |   | |   | |   | |   | |   | |
| (      | (\ (   | (            ) |   | |   | |   ) |   | |   | |   | |
| )      | ) \ \__| (____/\/\____) |___) (___| (__/  )___) (___| (___) |
|/       |/   \__/(_______/\_______)\_______/(______/ \_______/(_______)
"""


class Server:
    """Flask server for anonymizer."""

    def __init__(self):
        fileConfig(Path(Path(__file__).parent, LOGGING_CONF_FILE))
        self.logger = logging.getLogger("presidio-anonymizer")
        self.logger.setLevel(os.environ.get("LOG_LEVEL", self.logger.level))
        self.app = Flask(__name__)
        self.logger.info("Starting anonymizer engine")
        self.anonymizer = AnonymizerEngine()
        self.deanonymize = DeanonymizeEngine()
        self.logger.info(WELCOME_MESSAGE)

        @self.app.route("/health")
        def health() -> str:
            """Return basic health probe result."""
            return "Presidio Anonymizer service is up"

        @self.app.route("/anonymize", methods=["POST"])
        def anonymize() -> Response:
            content = request.get_json()
            if not content:
                raise BadRequest("Invalid request json")

            anonymizers_config = AppEntitiesConvertor.operators_config_from_json(
                content.get("anonymizers")
            )
            if AppEntitiesConvertor.check_custom_operator(anonymizers_config):
                raise BadRequest("Custom type anonymizer is not supported")

            text_raw = content.get("text", "")
            analyzer_raw = content.get("analyzer_results")

            if isinstance(text_raw, list):
                if not isinstance(analyzer_raw, list):
                    raise BadRequest(
                        "When 'text' is a list (batch), 'analyzer_results' must "
                        "be a list with one element per text — each element is the "
                        "analyzer output list for that document."
                    )
                if len(text_raw) != len(analyzer_raw):
                    raise BadRequest(
                        "When using batch anonymization, 'text' and "
                        f"'analyzer_results' must have the same length "
                        f"({len(text_raw)} != {len(analyzer_raw)})."
                    )
                batch_payload = []
                for i, piece in enumerate(text_raw):
                    ar_piece = analyzer_raw[i]
                    if ar_piece is None:
                        ar_piece = []
                    if not isinstance(ar_piece, list):
                        raise BadRequest(
                            "Each batch 'analyzer_results' entry must be a list "
                            "of recognition objects (same shape as the analyzer "
                            "returns for one text)."
                        )
                    analyzer_results = AppEntitiesConvertor.analyzer_results_from_json(
                        ar_piece
                    )
                    engine_result = self.anonymizer.anonymize(
                        text=str(piece),
                        analyzer_results=analyzer_results,
                        operators=anonymizers_config,
                    )
                    batch_payload.append(json.loads(engine_result.to_json()))
                return Response(
                    json.dumps(batch_payload), mimetype="application/json"
                )

            analyzer_results = AppEntitiesConvertor.analyzer_results_from_json(
                analyzer_raw
            )
            anoymizer_result = self.anonymizer.anonymize(
                text=text_raw if isinstance(text_raw, str) else str(text_raw),
                analyzer_results=analyzer_results,
                operators=anonymizers_config,
            )
            return Response(anoymizer_result.to_json(), mimetype="application/json")

        @self.app.route("/deanonymize", methods=["POST"])
        def deanonymize() -> Response:
            content = request.get_json()
            if not content:
                raise BadRequest("Invalid request json")
            text = content.get("text", "")
            deanonymize_entities = AppEntitiesConvertor.deanonymize_entities_from_json(
                content
            )
            deanonymize_config = AppEntitiesConvertor.operators_config_from_json(
                content.get("deanonymizers")
            )
            deanonymized_response = self.deanonymize.deanonymize(
                text=text, entities=deanonymize_entities, operators=deanonymize_config
            )
            return Response(
                deanonymized_response.to_json(), mimetype="application/json"
            )

        @self.app.route("/anonymizers", methods=["GET"])
        def anonymizers():
            """Return a list of supported anonymizers."""
            return jsonify(self.anonymizer.get_anonymizers())

        @self.app.route("/deanonymizers", methods=["GET"])
        def deanonymizers():
            """Return a list of supported deanonymizers."""
            return jsonify(self.deanonymize.get_deanonymizers())

        @self.app.errorhandler(InvalidParamError)
        def invalid_param(err):
            self.logger.warning(
                f"Request failed with parameter validation error: {err.err_msg}"
            )
            return jsonify(error=err.err_msg), 422

        @self.app.errorhandler(HTTPException)
        def http_exception(e):
            return jsonify(error=e.description), e.code

        @self.app.errorhandler(Exception)
        def server_error(e):
            self.logger.error(f"A fatal error occurred during execution: {e}")
            return jsonify(error="Internal server error"), 500

def create_app(): # noqa
    server = Server()
    return server.app

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    app.run(host="0.0.0.0", port=port)
