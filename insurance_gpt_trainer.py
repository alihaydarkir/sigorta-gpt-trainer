import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import os
from tqdm import tqdm
import re

# ============ OPTİMİZE KONFİGÜRASYON ============

class Config:
    """Temiz ve basit config"""
    model_name = "ytu-ce-cosmos/turkish-gpt2"
    
    # OPTİMİZE AYARLAR:
    max_length = 128
    batch_size = 2 if torch.cuda.is_available() else 1
    learning_rate = 3e-5
    num_epochs = 5
    gradient_accumulation_steps = 16

    # Learning rate scheduling
    warmup_steps = 50
    max_steps = 1000
    
    # Regularization
    weight_decay = 0.01

    # Dosyalar
    json_file = "sigorta_qa_ft.json"  # Değiştirildi - GitHub için
    output_dir = "clean_insurance_model"  # Değiştirildi - GitHub için
    
    # Cihaz otomatik seçimi
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# ============ VERİ TEMİZLİĞİ ============

def clean_text(text):
    """Metni temizle"""
    text = re.sub(r'[^\w\s.,;:!?\-ğüşıöçĞÜŞİÖÇ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class EnhancedCleanDataset(Dataset):
    """Zenginleştirilmiş dataset - DAHA İYİ ÖĞRENME İÇİN"""
    
    def __init__(self, json_file, tokenizer, max_length=128):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print("🧹 Veri temizleniyor ve zenginleştiriliyor...")
        cleaned_data = []
        
        for item in self.data:
            clean_q = clean_text(item['input'])
            clean_a = clean_text(item['output'])
            
            if len(clean_a) > 10 and len(clean_q) > 5:
                # Çeşitli formatlarda veri ekle (data augmentation)
                formats = [
                    f"Soru: {clean_q}\nCevap: {clean_a}",
                    f"Q: {clean_q}\nA: {clean_a}",
                    f"{clean_q}?\n{clean_a}",
                    f"### Soru: {clean_q}\n### Cevap: {clean_a}"
                ]
                
                for formatted_text in formats:
                    cleaned_data.append({
                        'text': formatted_text,
                        'input': clean_q,
                        'output': clean_a
                    })
        
        self.data = cleaned_data
        print(f"✅ {len(self.data)} zenginleştirilmiş örnek hazır")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

class CleanDataset(Dataset):
    """Temizlenmiş dataset - DÜZELTİLMİŞ"""
    
    def __init__(self, json_file, tokenizer, max_length=128):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print("🧹 Veri temizleniyor...")
        cleaned_data = []
        for item in self.data:
            clean_q = clean_text(item['input'])
            clean_a = clean_text(item['output'])
            
            if len(clean_a) > 10 and len(clean_q) > 5:
                cleaned_data.append({
                    'input': clean_q,
                    'output': clean_a
                })
        
        self.data = cleaned_data
        print(f"✅ {len(self.data)} temiz örnek hazır")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # BASİT VE GÜVENLİ FORMAT
        text = f"Soru: {item['input']}\nCevap: {item['output']}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

# ============ KONTROLLÜ EĞİTİM ============

def controlled_training(model, tokenizer, dataset, config):
    """Kontrollü eğitim"""
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    accumulation_steps = config.gradient_accumulation_steps
    
    # Loss takibi için
    loss_values = []
    
    print("🎯 Kontrollü eğitim başlıyor...")
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * accumulation_steps
            
            # Loss kaydetme
            loss_values.append(loss.item() * accumulation_steps)
            
            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()
            
            # Her 10 step'te ortalama loss göster
            if step % 10 == 0 and len(loss_values) >= 10:
                avg_recent_loss = sum(loss_values[-10:]) / 10
                pbar.set_postfix({
                    'loss': f'{(loss.item() * accumulation_steps):.4f}',
                    'avg_10': f'{avg_recent_loss:.4f}'
                })
            else:
                pbar.set_postfix({'loss': f'{(loss.item() * accumulation_steps):.4f}'})
            
            if loss.item() * accumulation_steps > 10:
                print("⚠️ Loss çok yüksek, eğitim durduruluyor!")
                return
        
        if len(dataloader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        test_model(model, tokenizer, config.device)
    
    # Loss grafiği (opsiyonel)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(loss_values)
        plt.title('Loss Değerleri')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.savefig('loss_graph.png')
        print("📊 Loss grafiği kaydedildi: loss_graph.png")
    except:
        print("ℹ️ Loss grafiği için matplotlib gerekli")

# ============ TEST FONKSİYONLARI ============

def test_model(model, tokenizer, device):
    """Modeli test et - İYİLEŞTİRİLMİŞ"""
    model.eval()
    
    test_q = "Aktüerler Sicili kimde tutulur?"
    prompt = f"Soru: {test_q}\nCevap:"
    
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=80,
            temperature=0.8,
            do_sample=True,
            top_k=45,
            top_p=0.95,
            repetition_penalty=1.4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.replace(prompt, "").strip()
    
    print(f"Test Q: {test_q}")
    print(f"Test A: {answer[:100]}")
    print("-" * 50)

def comprehensive_test(model, tokenizer, device):
    """Kapsamlı test - YENİ"""
    print("\n🧪 KAPSAMLI TEST:")
    print("="*60)
    
    test_cases = [
        "Sigorta poliçesi nedir?",
        "Broker kimdir?",
        "Minimum garanti fonu nedir?",
        "Aktüer ne iş yapar?",
        "Reasürans nedir?",
        "Sigorta acentesi ile broker arasındaki fark nedir?"
    ]
    
    for i, question in enumerate(test_cases, 1):
        prompt = f"Soru: {question}\nCevap:"
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=100,
                temperature=0.8,
                do_sample=True,
                top_k=45,
                top_p=0.95,
                repetition_penalty=1.4,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.replace(prompt, "").strip()
        
        print(f"\n{i}. 🔵 {question}")
        print(f"   🟢 {answer}")
        print("   " + "─" * 50)

def quality_test(model, tokenizer, device):
    """Model kalitesini test et - DÜZELTİLMİŞ"""
    print("📊 Model Kalite Metrikleri:")
    print("="*40)
    
    # DAHA UYGUN ANAHTAR KELİMELER
    questions = [
        ("Sigorta poliçesi nedir?", "sigorta"),
        ("Broker kimdir?", "broker"),
        ("Aktüer ne iş yapar?", "aktüer")
    ]
    correct_answers = 0
    
    for question, keyword in questions:
        prompt = f"Soru: {question}\nCevap:"
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=60,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.replace(prompt, "").strip().lower()
        
        # Anahtar kelime kontrolü
        if keyword.lower() in answer:
            correct_answers += 1
            print(f"✅ {question} -> Doğru ('{keyword}' bulundu)")
        else:
            print(f"❌ {question} -> Beklenen: '{keyword}', Alınan: '{answer[:50]}...'")
    
    accuracy = (correct_answers / len(questions)) * 100
    print(f"\n📈 Doğruluk: {accuracy:.1f}%")

# ============ MAIN FONKSİYON ============

def main():
    """Ana fonksiyon - GitHub için düzeltildi"""
    
    print("🚨 SİGORTA GPT EĞİTİMİ")
    print("="*50)
    
    # ADIM 1: TEMİZ MODEL VE TOKENIZER
    print("1️⃣ Model yükleniyor...")
    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(config.model_name)
    
    # Dropout ayarları
    print("⚙️ Dropout ayarlanıyor...")
    if hasattr(model.config, 'resid_pdrop'):
        model.config.resid_pdrop = 0.2
    if hasattr(model.config, 'embd_pdrop'):
        model.config.embd_pdrop = 0.2
    if hasattr(model.config, 'attn_pdrop'):
        model.config.attn_pdrop = 0.2
    
    model.to(config.device)
    
    print(f"✅ Model: {config.model_name}")
    print(f"✅ Cihaz: {config.device}")
    print(f"✅ Vocab size: {len(tokenizer)}")
    
    # ADIM 2: VERİ YÜKLENİYOR
    print("\n2️⃣ Veri yükleniyor...")
    try:
        dataset = CleanDataset(config.json_file, tokenizer, config.max_length)
    except FileNotFoundError:
        print(f"❌ Dosya bulunamadı: {config.json_file}")
        print("📋 Lütfen 'sigorta_qa_ft.json' dosyasını ana dizine koyun")
        return
    
    # ADIM 3: TEST - EĞİTİM ÖNCESİ
    print("\n3️⃣ Eğitim öncesi test:")
    test_model(model, tokenizer, config.device)
    
    # ADIM 4: EĞİTİM
    print("\n4️⃣ Eğitim başlıyor...")
    controlled_training(model, tokenizer, dataset, config)
    
    # ADIM 5: KAYDET
    print("\n5️⃣ Model kaydediliyor...")
    os.makedirs(config.output_dir, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"✅ Kaydedildi: {config.output_dir}")
    
    # ADIM 6: KAPSAMLI TEST
    print("\n6️⃣ Kapsamlı test:")
    comprehensive_test(model, tokenizer, config.device)
    
    # ADIM 7: KALİTE TESTİ
    print("\n7️⃣ Kalite testi:")
    quality_test(model, tokenizer, config.device)
    
    print("\n🎉 EĞİTİM TAMAMLANDI!")

if __name__ == "__main__":
    main()
