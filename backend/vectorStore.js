const { OpenAI } = require('openai');
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

let documents = []; // In-memory storage

async function embedText(text) {
    const res = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
    });
    return res.data[0].embedding;
}

function cosineSimilarity(vecA, vecB) {
    const dot = vecA.reduce((acc, val, i) => acc + val * vecB[i], 0);
    const magA = Math.sqrt(vecA.reduce((acc, val) => acc + val * val, 0));
    const magB = Math.sqrt(vecB.reduce((acc, val) => acc + val * val, 0));
    return dot / (magA * magB);
}

async function addDocument(text) {
    const embedding = await embedText(text);
    documents.push({ text, embedding });
}

async function searchSimilar(query, topK = 3) {
    const queryEmbedding = await embedText(query);
    return documents
        .map(doc => ({
            ...doc,
            score: cosineSimilarity(queryEmbedding, doc.embedding)
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, topK);
}

module.exports = { addDocument, searchSimilar };
