import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  CollectionItem,
  DocumentItem,
  DebugResult,
  EvalDatasetItem,
  EvalResponse,
  ExplorerData,
} from '../models/rag.models';

@Injectable({
  providedIn: 'root'
})
export class RagApiService {
  private baseUrl = 'http://localhost:8000/api';

  constructor(private http: HttpClient) {}

  // Healthcheck
  getHealth(): Observable<any> {
    return this.http.get(`${this.baseUrl}/health`);
  }

  // Collections
  getCollections(): Observable<CollectionItem[]> {
    return this.http.get<CollectionItem[]>(`${this.baseUrl}/collections`);
  }

  createCollection(name: string): Observable<{ status: string; name: string }> {
    return this.http.post<{ status: string; name: string }>(`${this.baseUrl}/collections`, { name });
  }

  deleteCollection(name: string): Observable<any> {
    return this.http.delete(`${this.baseUrl}/collections/${encodeURIComponent(name)}`);
  }

  // Documents
  getDocuments(collectionName: string): Observable<DocumentItem[]> {
    return this.http.get<DocumentItem[]>(`${this.baseUrl}/documents`, {
      params: { collection_name: collectionName }
    });
  }

  uploadDocuments(files: File[], collectionName: string, preventDuplicates: boolean = true): Observable<any> {
    const formData = new FormData();
    files.forEach(f => formData.append('files', f, f.name));
    formData.append('collection_name', collectionName);
    formData.append('prevent_duplicates', String(preventDuplicates));
    return this.http.post(`${this.baseUrl}/documents/upload`, formData);
  }

  deleteDocument(filename: string, collectionName: string): Observable<any> {
    return this.http.delete(`${this.baseUrl}/documents/${encodeURIComponent(filename)}`, {
      params: { collection_name: collectionName }
    });
  }

  updateDocument(file: File, targetFilename: string, collectionName: string): Observable<any> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    formData.append('target_filename', targetFilename);
    formData.append('collection_name', collectionName);
    return this.http.post(`${this.baseUrl}/documents/update`, formData);
  }

  // Chat & Q&A
  chat(payload: {
    collection_name: string;
    query: string;
    mode: string;
    use_query_rewriting?: boolean;
    use_hybrid?: boolean;
    use_reranking?: boolean;
    use_compression?: boolean;
    k?: number;
  }): Observable<{ answer: string; docs: any[]; telemetry: any }> {
    return this.http.post<{ answer: string; docs: any[]; telemetry: any }>(`${this.baseUrl}/chat`, payload);
  }

  transcribeAudio(file: File): Observable<{ transcript: string }> {
    const formData = new FormData();
    formData.append('audio', file, file.name);
    return this.http.post<{ transcript: string }>(`${this.baseUrl}/transcribe-audio`, formData);
  }

  // Advanced Retrieval Diagnostics
  retrieveDebug(payload: {
    collection_name: string;
    query: string;
    search_type: string;
    k: number;
    fetch_k?: number;
    lambda_mult?: number;
    score_threshold?: number | null;
    metadata_filter?: any;
  }): Observable<DebugResult> {
    return this.http.post<DebugResult>(`${this.baseUrl}/retrieve/debug`, payload);
  }

  // RAG Evaluation
  generateEvalDataset(collectionName: string, numQuestions: number = 3): Observable<{ dataset: EvalDatasetItem[] }> {
    return this.http.post<{ dataset: EvalDatasetItem[] }>(`${this.baseUrl}/eval/generate-dataset`, {
      collection_name: collectionName,
      num_questions: numQuestions
    });
  }

  runEvaluation(collectionName: string, dataset: EvalDatasetItem[], engine: string = 'standard'): Observable<EvalResponse> {
    return this.http.post<EvalResponse>(`${this.baseUrl}/eval/run`, {
      collection_name: collectionName,
      dataset,
      engine
    });
  }

  // ChromaDB Explorer
  getExplorerData(collectionName: string): Observable<ExplorerData> {
    return this.http.get<ExplorerData>(`${this.baseUrl}/explorer`, {
      params: { collection_name: collectionName }
    });
  }
}

