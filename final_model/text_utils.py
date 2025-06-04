import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stopword_factory = StopWordRemoverFactory()
stemmer_factory = StemmerFactory()
stopwords = set(stopword_factory.get_stop_words())
stemmer = stemmer_factory.create_stemmer()

informal_dict = {
    'gak': 'tidak', 'ga': 'tidak', 'nggak': 'tidak',
    'aja': 'saja', 'udah': 'sudah', 'ngga': 'tidak',
    'kalo': 'kalau'
}

def clean_text(text, stopwords=stopwords, informal_dict=informal_dict):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    words = text.split()
    words = [informal_dict.get(word, word) for word in words]
    words = [word for word in words if word not in stopwords]
    text = ' '.join(words)
    text = stemmer.stem(text)
    return text

def tail_tokenizer(text, tokenizer, max_length=128):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=True,
        return_token_type_ids=False
    )
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    if len(input_ids) <= max_length:
        padded = tokenizer.pad(
            {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask]
            },
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
    else:
        input_ids = input_ids[-max_length:]
        attention_mask = attention_mask[-max_length:]
        padded = tokenizer.pad(
            {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask]
            },
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

    return padded