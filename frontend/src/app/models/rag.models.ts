export interface CollectionItem {
  name: string;
  count: number;
  id?: string;
}

export interface DocumentItem {
  doc_id: string;
  filename: string;
  file_hash: string;
  file_size: number;
  total_pages: number;
  upload_time: string;
  chunk_count: number;
}

export interface SourceChunk {
  filename: string;
  page: number;
  chunk_id: string;
  content: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceChunk[];
  telemetry?: any;
  timestamp?: string;
}

export interface DebugItem {
  rank: number;
  score: number;
  raw_score: number;
  filename: string;
  page: number;
  total_pages: number | string;
  chunk_id: string;
  content_length: number;
  content_preview: string;
  full_content: string;
  metadata: any;
}

export interface DebugResult {
  query: string;
  search_type: string;
  k: number;
  fetch_k?: number;
  lambda_mult?: number;
  score_threshold?: number;
  filter?: any;
  latency_ms: number;
  num_retrieved: number;
  items: DebugItem[];
}

export interface EvalDatasetItem {
  id: string;
  question: string;
  ground_truth_answer: string;
  source_chunk_id?: string;
  source_filename?: string;
  source_page?: number;
  source_context?: string;
}

export interface EvalSummary {
  mean_recall_at_k: number;
  mean_context_relevance: number;
  mean_faithfulness: number;
  mean_answer_relevance: number;
  mean_token_f1: number;
  total_evaluated: number;
}

export interface EvalItemResult {
  id: string;
  question: string;
  ground_truth: string;
  predicted_answer: string;
  recall_at_k: number;
  context_relevance: number;
  faithfulness: number;
  answer_relevance: number;
  token_f1: number;
  retrieved_count: number;
}

export interface EvalResponse {
  summary: EvalSummary;
  results: EvalItemResult[];
}

export interface PcaPoint {
  chunk_id: string;
  x: number;
  y: number;
  filename: string;
  page: number;
}

export interface ExplorerData {
  collection_name: string;
  total_chunks: number;
  embedding_dim: string | number;
  chunks: {
    chunk_id: string;
    filename: string;
    page: number;
    length: number;
    content: string;
    metadata: any;
  }[];
  pca_points: PcaPoint[];
}

