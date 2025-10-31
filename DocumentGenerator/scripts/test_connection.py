import os
import requests
from openai import AzureOpenAI
from dotenv import load_dotenv

# Carrega variáveis do .env
load_dotenv()

# Configuração
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_version = "2023-05-15"

print("=== Teste de Conexão Azure OpenAI ===")
print(f"Endpoint: {endpoint}")
print(f"Deployment: {deployment_name}")
print(f"API Version: {api_version}")
print(f"API Key: {'*' * 10}{api_key[-4:] if api_key else 'NOT SET'}")
print()

# Teste 1: Verificar se o endpoint responde
print("1. Testando conectividade básica ao endpoint...")
try:
    response = requests.get(endpoint, timeout=10)
    print(f"✓ Endpoint acessível (Status: {response.status_code})")
except requests.exceptions.RequestException as e:
    print(f"✗ Erro de conectividade: {e}")
    print("  - Verifique sua conexão à internet")
    print("  - Verifique se o endpoint está correto")

# Teste 2: Testar listagem de deployments
print("\n2. Testando autenticação e listagem de deployments...")
try:
    deployments_url = f"{endpoint}openai/deployments?api-version={api_version}"
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    
    response = requests.get(deployments_url, headers=headers, timeout=10)
    
    if response.status_code == 200:
        deployments = response.json()
        print("✓ Autenticação bem-sucedida!")
        print("Deployments disponíveis:")
        for deployment in deployments.get("data", []):
            print(f"  - {deployment.get('id', 'N/A')} (modelo: {deployment.get('model', 'N/A')})")
        
        # Verificar se o deployment especificado existe
        deployment_ids = [d.get('id') for d in deployments.get("data", [])]
        if deployment_name in deployment_ids:
            print(f"✓ Deployment '{deployment_name}' encontrado!")
        else:
            print(f"✗ Deployment '{deployment_name}' NÃO encontrado!")
            print(f"  Deployments disponíveis: {deployment_ids}")
    else:
        print(f"✗ Erro na autenticação (Status: {response.status_code})")
        print(f"  Resposta: {response.text}")
        
except requests.exceptions.RequestException as e:
    print(f"✗ Erro de rede: {e}")
except Exception as e:
    print(f"✗ Erro inesperado: {e}")

# Teste 3: Testar o cliente OpenAI
print("\n3. Testando cliente Azure OpenAI...")
try:
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": "Diga apenas 'Olá' para testar a conexão."}],
        max_tokens=10,
        temperature=0
    )
    
    print("✓ Cliente OpenAI funcionando!")
    print(f"  Resposta: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"✗ Erro no cliente OpenAI: {e}")
    print("  Possíveis causas:")
    print("  - Deployment não existe ou não está ativo")
    print("  - Problemas de quota ou rate limiting")
    print("  - Chave API inválida ou expirada")

print("\n=== Fim do teste ===")