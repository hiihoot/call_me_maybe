# from pydantic import BaseModel, Field
# from typing import List

# class User(BaseModel):
#     name: str
#     age: int = Field(ge=0, le=150)  # Between 0 and 150
#     email: str
#     skills: List[str] = []

# # Generate JSON schema
# schema = User.model_json_schema()

# # Validate data
# user = User(name="Alice", age=30, email="alice@example.com")

import json



res = {
    "prompt": "what is X",
    "name": "fn_this_func",
    "parameters": 1
}

r = json.dumps(res)
print(r)