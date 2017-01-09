schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "properties": {
        "objects": {
            "type": "object",
            "properties": {
                "global_ID": {
                    "type": "object",
                    "properties": {
                        "objectClass": {
                            "type": "string"
                        },
                        "globalID": {
                            "type": "string"
                        },
                        "rectangle": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "classDescription": {
                            "type": "string"
                        },
                        "shortName": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "objectClass",
                        "globalID",
                        "rectangle",
                        "classDescription",
                        "shortName"
                    ]
                },
                "questions": {
                    "qid": {
                        "type": "string"
                    },
                    "questionText": {
                        "type": "string"
                    },
                    "questionType": {
                        "type": "string"
                    },
                    "objectsInvolved": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "objectTypesInvolved": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                }
            }
        }
    }
}