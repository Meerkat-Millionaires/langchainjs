import { createClient } from "redis";
import { OpenAIEmbeddings } from "./embeddings/openai.js";
import { RedisVectorStore } from "./vectorstores/solami.js";
import { DirectoryLoader, TextLoader } from "./document_loaders/index.js";
import { RecursiveCharacterTextSplitter } from "./text_splitter.js";
import { VectorDBQAChain } from "./chains/vector_db_qa.js";
import { ChainTool } from "./tools/chain.js";
import { OpenAI } from "./llms/openai.js";
import { initializeAgentExecutorWithOptions } from "./agents/initialize.js";
const client = createClient({
  url: process.env.REDIS_URL ?? "redis://redis-10191.c27035.us-east-1-mz.ec2.cloud.rlrcp.com:10191",
  username: "default",
  password: "r5JSXI9XNO3rlzeGgcvVB4PlkQeOP1qe"
});
async function doit(){
await client.connect();
let loader = new DirectoryLoader("./csv",  {
  ".txt": (path) => new TextLoader(path),
}, true);
let docs = await loader.load();

const docSplitter = new RecursiveCharacterTextSplitter()
docs = await docSplitter.splitDocuments(docs);

const vectorStore = await RedisVectorStore.fromDocuments(
  docs,
  new OpenAIEmbeddings(),
  {
    redisClient: client,
    indexName: "docs3",
  }
);
const model = new OpenAI({ temperature: 0.2 });
const chain = VectorDBQAChain.fromLLM(model, vectorStore);

const qaTool = new ChainTool({
  name: "state-of-union-qa",
  description:
    "Your context is conversations Lesley has had with people. Ask Lesley about anything. She will answer with her opinion and some length contextual content.",
  chain: chain,
});

const tools = [
  /*new SerpAPI(process.env.SERPAPI_API_KEY, {
    location: "Austin,Texas,United States",
    hl: "en",
    gl: "us",
  }),
  new Calculator(), */
  qaTool,
];

const executor = await initializeAgentExecutorWithOptions(tools, model, {
  agentType: "zero-shot-react-description",
});
console.log("Loaded agent.");

let inputs = ["How are you, Lesley?",
"Who are your children?",
"What do you thnk of Trudeau?",
"What do you think of Jarett?",
"What do you think of Tiana?",
"What do you think of Damion?",
"What do you think of Reggie?"]
for (var input of inputs){
const result = await executor.call({ input });

console.log(`${input}\n ${result.output} \n\n`);
}


await client.disconnect();
}
doit();