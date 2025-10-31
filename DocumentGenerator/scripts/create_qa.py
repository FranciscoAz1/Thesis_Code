import json
import random
import os
from openai import OpenAI
from docx import Document
from dotenv import load_dotenv

# Carrega variáveis do .env
load_dotenv()

# Configuração do cliente OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Verificação das variáveis obrigatórias
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("A variável OPENAI_API_KEY não está definida no .env")

print(f"Usando modelo: {model_name}")

def read_docx_content(file_path):
    """
    Lê o conteúdo de um arquivo DOCX e retorna o texto completo.
    """
    try:
        doc = Document(file_path)
        content = []
        
        # Extrai todos os parágrafos
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                content.append(paragraph.text.strip())
        
        # Junta todo o conteúdo
        full_content = '\n'.join(content)
        return full_content
    except Exception as e:
        print(f"Erro ao ler arquivo {file_path}: {str(e)}")
        return None

def generate_qa_with_gpt(document_content, file_name):
    """
    Gera uma pergunta e resposta baseada no conteúdo do documento usando OpenAI.
    """
    prompt = f"""
Baseado no seguinte documento administrativo, crie uma pergunta específica que REQUEIRA as informações contidas no documento para ser respondida corretamente.

DOCUMENTO:
{document_content}

INSTRUÇÕES:
1. A pergunta deve ser específica e contextual - deve exigir informações ÚNICAS deste documento
2. A pergunta deve ser natural e realista - como se alguém estivesse procurando informações específicas
3. A resposta deve ser precisa e baseada EXCLUSIVAMENTE no conteúdo fornecido
4. Evite perguntas genéricas que poderiam ser respondidas sem ler o documento

FORMATO DE RESPOSTA (JSON):
{{
    "pergunta": "sua pergunta específica aqui",
    "resposta": "sua resposta detalhada baseada no documento"
}}

Responda APENAS com o JSON válido, sem texto adicional.
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
            service_tier="flex"
        )
        
        # Tenta fazer parse do JSON da resposta
        response_text = response.choices[0].message.content.strip()
        
        # Remove possíveis caracteres extras antes e depois do JSON
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        qa_data = json.loads(response_text)
        return qa_data
        
    except json.JSONDecodeError as e:
        print(f"Erro ao fazer parse JSON para {file_name}: {str(e)}")
        print(f"Resposta recebida: {response_text}")
        return None
    except Exception as e:
        print(f"Erro ao gerar Q&A para {file_name}: {str(e)}")
        return None

def select_random_files(directory, n=20):
    """
    Seleciona n arquivos aleatórios da pasta de documentos.
    """
    try:
        all_files = [f for f in os.listdir(directory) if f.endswith('.docx')]
        if len(all_files) < n:
            print(f"Aviso: Apenas {len(all_files)} arquivos encontrados, usando todos.")
            return all_files
        
        selected_files = random.sample(all_files, n)
        return selected_files
    except Exception as e:
        print(f"Erro ao selecionar arquivos: {str(e)}")
        return []

def create_qa_dataset(documents_dir="documentos_gerados", output_file="datasets/qa_dataset.json", n_files=20):
    """
    Cria um dataset de Q&A a partir dos documentos DOCX.
    """
    print(f"Iniciando criação do dataset Q&A...")
    print(f"Diretório: {documents_dir}")
    print(f"Número de arquivos: {n_files}")
    
    # Seleciona arquivos aleatórios
    selected_files = select_random_files(documents_dir, n_files)
    
    if not selected_files:
        print("Nenhum arquivo selecionado. Encerrando.")
        return
    
    print(f"Arquivos selecionados: {len(selected_files)}")
    for i, file in enumerate(selected_files, 1):
        print(f"  {i}. {file}")
    
    qa_dataset = []
    
    for i, filename in enumerate(selected_files, 1):
        file_path = os.path.join(documents_dir, filename)
        print(f"\n[{i}/{len(selected_files)}] Processando: {filename}")
        
        # Lê o conteúdo do documento
        content = read_docx_content(file_path)
        if not content:
            print(f"  ❌ Falha ao ler conteúdo")
            continue
        
        print(f"  📄 Conteúdo extraído: {len(content)} caracteres")
        
        # Gera pergunta e resposta
        qa_data = generate_qa_with_gpt(content, filename)
        if not qa_data:
            print(f"  ❌ Falha ao gerar Q&A")
            continue
        
        # Adiciona ao dataset
        dataset_entry = {
            "arquivo": filename,
            "contexto": content,
            "pergunta": qa_data.get("pergunta", ""),
            "resposta": qa_data.get("resposta", "")
        }
        
        qa_dataset.append(dataset_entry)
        print(f"  ✅ Q&A gerado com sucesso")
        print(f"     Pergunta: {qa_data.get('pergunta', '')[:100]}...")
    
    # Salva o dataset
    if qa_dataset:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_dataset, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 Dataset criado com sucesso!")
        print(f"📊 Total de entradas: {len(qa_dataset)}")
        print(f"💾 Arquivo salvo: {output_file}")
        
        # Mostra uma amostra
        if qa_dataset:
            print(f"\n📋 Exemplo do primeiro item:")
            first_item = qa_dataset[0]
            print(f"   Arquivo: {first_item['arquivo']}")
            print(f"   Pergunta: {first_item['pergunta']}")
            print(f"   Resposta: {first_item['resposta'][:200]}...")
    else:
        print("\n❌ Nenhum item foi adicionado ao dataset.")

if __name__ == "__main__":
    # Cria o dataset com 20 arquivos aleatórios
    create_qa_dataset(
        documents_dir="documentos_gerados",
        output_file="datasets/qa_dataset300.json",
        n_files=300
    )
