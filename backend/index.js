require('dotenv').config();
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const { OpenAI } = require('openai');
const { extractTextFromPDF, chunkText } = require('./pdfProcessor');
const { addDocument, searchSimilar } = require('./vectorStore');

const app = express();
app.use(cors());
app.use(bodyParser.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const upload = multer({ dest: 'uploads/' });

app.post('/upload', upload.single('file'), async (req, res) => {
    const filePath = req.file.path;
    const text = await extractTextFromPDF(filePath);
    const chunks = chunkText(text);
    for (const chunk of chunks) {
        await addDocument(chunk);
    }
    fs.unlinkSync(filePath);
    res.json({ message: 'PDF processed and indexed.' });
});

app.post('/ask', async (req, res) => {
    const { question } = req.body;
    const relevantChunks = await searchSimilar(question);
    const context = relevantChunks.map(c => c.text).join('\n\n');

    const response = await openai.chat.completions.create({
        model: 'gpt-4',
        messages: [
            { role: 'system', content: 'Answer based on the context below.' },
            { role: 'user', content: `${context}\n\nQuestion: ${question}` }
        ],
    });

    // const response = {
    //     choices: [
    //       {
    //         message: {
    //           content: `🧪 FAKE ANSWER for: "${question}"\n(Top Context: "${context.slice(0, 100)}...")`
    //         }
    //       }
    //     ]
    //   };

    res.json({ answer: response.choices[0].message.content });
});

app.listen(process.env.PORT, () => {
    console.log(`Server running on port ${process.env.PORT}`);
});
