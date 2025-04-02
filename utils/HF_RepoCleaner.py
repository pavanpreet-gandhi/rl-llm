from huggingface_hub import HfApi

api = HfApi()
username = api.whoami()["name"]
print(f"Logged in as {username}")

# # Uncomment to DELETE all models /!\
# models = list(api.list_models(author=username))
# # models = [i for i in models if "2025-03" in i.modelId]
# # breakpoint()
# print(models)
# print(f"Found {len(models)} models:")
# for model in models:
#     api.delete_repo(model.id)
#     print(f"Deleted {model.id}")
