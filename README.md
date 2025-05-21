# RAG aplikacija

## Programos aprašymas

Ši programa yra RAG (Retrieval-Augmented Generation) tipo aplikacija, skirta atsakyti į klausimus apie PDF dokumentą.

### Kaip veikia

1. **Duomenų apdorojimas:**  
   Programa įkelia PDF failą (`python_ivadas.pdf`) ir suskaido jį į mažesnius teksto fragmentus naudodama `RecursiveCharacterTextSplitter`. Tai leidžia efektyviau apdoroti ir ieškoti informacijos.

2. **Paieška (vektorinė paieška):**  
   Kiekvienas teksto fragmentas paverčiamas vektoriais su `HuggingFaceEmbeddings` modeliu. Šie vektoriai saugomi Chroma vektorinėje duomenų bazėje. Užklausos metu ieškoma panašiausių fragmentų pagal klausimą.

3. **Atsakymo formavimas:**  
   Rasta informacija (kontekstas) perduodama generatyviam LLM modeliui (`ChatGoogleGenerativeAI` su Gemini API). Prieš atsakant, modelis visada nuramina vartotoją, kad mokytis yra saugu, ir tik tada pateikia atsakymą pagal rastą kontekstą.

### Sprendimo struktūra

- PDF įkėlimas ir suskaidymas į fragmentus
- Fragmentų pavertimas vektoriais ir saugojimas
- Klausimo analizė ir panašiausių fragmentų paieška
- Atsakymo generavimas pagal rastą kontekstą ir specialų šabloną

### Pagrindiniai komponentai

- `PyPDFLoader` (PDF įkėlimui)
- `RecursiveCharacterTextSplitter` (teksto dalinimui)
- `HuggingFaceEmbeddings` ir `Chroma` (vektorinei paieškai)
- `ChatGoogleGenerativeAI` (atsakymo generavimui)
- `PromptTemplate` (atsakymo šablonui)

### Reikalingas .env failas
Norint naudoti šią programą, reikia sukurti `.env` failą su šiais kintamaisiais:

```GOOGLE_API_KEY=your_google_api_key```