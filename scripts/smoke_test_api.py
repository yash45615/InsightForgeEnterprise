from fastapi.testclient import TestClient

from insightforge.main import app


def main() -> None:
    client = TestClient(app)

    response = client.get("/openapi.json")
    assert response.status_code == 200, f"OpenAPI returned HTTP {response.status_code}"

    schema = response.json()
    assert isinstance(schema.get("paths"), dict), (
        "OpenAPI schema does not contain a valid paths object"
    )

    paths = schema["paths"]
    assert paths, "No API paths are registered"

    print(f"Application: {schema.get('info', {}).get('title', app.title)}")
    print(f"Version: {schema.get('info', {}).get('version', app.version)}")
    print(f"Registered paths: {len(paths)}")

    for path in sorted(paths):
        methods = [
            method.upper()
            for method in paths[path]
            if method.lower()
            in {"get", "post", "put", "patch", "delete", "options", "head"}
        ]
        print(f"{','.join(sorted(methods)):20} {path}")

    docs_response = client.get("/docs")
    assert docs_response.status_code == 200, (
        f"Swagger UI returned HTTP {docs_response.status_code}"
    )

    print("\nAPI smoke test passed.")


if __name__ == "__main__":
    main()
