import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RagApiService } from './services/rag-api.service';
import {
  CollectionItem,
  DocumentItem,
  ChatMessage,
  DebugResult,
  EvalDatasetItem,
  EvalResponse,
  ExplorerData,
  PcaPoint,
} from './models/rag.models';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'Intelligent PDF RAG';

  // Navigation Tabs
  activeTab: 'chat' | 'docs' | 'retrieval' | 'eval' | 'explorer' = 'chat';

  // Backend Health
  healthStatus: { status: string; llm_model?: string; embedding_model?: string } | null = null;

  // Collections
  collections: CollectionItem[] = [];
  activeCollection: string = 'default_pdf_rag';
  newCollectionName: string = '';
  isCreatingCollection: boolean = false;

  // TAB 1: Chat State
  messages: ChatMessage[] = [];
  queryText: string = '';
  ragMode: 'standard' | 'advanced' = 'standard';
  useQueryRewriting: boolean = true;
  useHybrid: boolean = true;
  useReranking: boolean = true;
  useCompression: boolean = true;
  isChatLoading: boolean = false;
  isTranscribing: boolean = false;
  expandedSourceMessageIndex: number | null = null;

  // TAB 2: Document Management State
  documents: DocumentItem[] = [];
  selectedFilesToUpload: File[] = [];
  preventDuplicates: boolean = true;
  isUploading: boolean = false;
  uploadAlert: { type: 'success' | 'info' | 'error'; message: string } | null = null;
  updateTargetDoc: string = '';
  updateFile: File | null = null;
  isUpdating: boolean = false;

  // TAB 3: Advanced Retrieval Diagnostics State
  debugQuery: string = 'What are the core technical concepts and architecture?';
  searchType: 'similarity' | 'mmr' | 'similarity_score_threshold' = 'similarity';
  retrievalK: number = 4;
  retrievalFetchK: number = 15;
  retrievalLambda: number = 0.5;
  retrievalScoreThreshold: number = 0.5;
  metadataFilterDoc: string = 'ALL';
  debugResult: DebugResult | null = null;
  isDebugLoading: boolean = false;
  selectedInspectRank: number = 1;

  // TAB 4: RAG Evaluation State
  numEvalQuestions: number = 3;
  evalDataset: EvalDatasetItem[] = [];
  isGeneratingDataset: boolean = false;
  evalEngine: 'standard' | 'advanced' = 'standard';
  isRunningEval: boolean = false;
  evalResponse: EvalResponse | null = null;

  // TAB 5: ChromaDB Explorer State
  explorerData: ExplorerData | null = null;
  isLoadingExplorer: boolean = false;
  chunkFilterKeyword: string = '';
  selectedInspectChunkId: string | null = null;

  constructor(private api: RagApiService) {}

  ngOnInit(): void {
    this.checkHealth();
    this.loadCollections();
  }

  // Health & Collections
  checkHealth(): void {
    this.api.getHealth().subscribe({
      next: (res) => (this.healthStatus = res),
      error: (err) => console.error('Health check failed:', err)
    });
  }

  loadCollections(): void {
    this.api.getCollections().subscribe({
      next: (cols) => {
        this.collections = cols;
        if (cols.length > 0 && (!this.activeCollection || !cols.some(c => c.name === this.activeCollection))) {
          this.activeCollection = cols[0].name;
        }
        this.onCollectionChanged();
      },
      error: (err) => console.error('Failed to load collections:', err)
    });
  }

  onCollectionChanged(): void {
    this.loadDocuments();
    if (this.activeTab === 'explorer') {
      this.loadExplorerData();
    }
  }

  createCollection(): void {
    if (!this.newCollectionName.trim()) return;
    this.isCreatingCollection = true;
    this.api.createCollection(this.newCollectionName.trim()).subscribe({
      next: (res) => {
        this.isCreatingCollection = false;
        this.activeCollection = res.name;
        this.newCollectionName = '';
        this.loadCollections();
      },
      error: (err) => {
        this.isCreatingCollection = false;
        alert('Failed to create collection: ' + (err.error?.detail || err.message));
      }
    });
  }

  deleteActiveCollection(): void {
    if (!confirm(`Are you sure you want to delete collection "${this.activeCollection}" and all its data?`)) return;
    this.api.deleteCollection(this.activeCollection).subscribe({
      next: () => {
        this.activeCollection = 'default_pdf_rag';
        this.loadCollections();
      },
      error: (err) => alert('Failed to delete collection: ' + (err.error?.detail || err.message))
    });
  }

  // TAB 1: Chat & Q&A
  sendMessage(): void {
    const q = this.queryText.trim();
    if (!q || this.isChatLoading) return;

    this.messages.push({
      role: 'user',
      content: q,
      timestamp: new Date().toLocaleTimeString()
    });
    this.queryText = '';
    this.isChatLoading = true;

    this.api.chat({
      collection_name: this.activeCollection,
      query: q,
      mode: this.ragMode,
      use_query_rewriting: this.useQueryRewriting,
      use_hybrid: this.useHybrid,
      use_reranking: this.useReranking,
      use_compression: this.useCompression,
      k: 4,
    }).subscribe({
      next: (res) => {
        this.isChatLoading = false;
        this.messages.push({
          role: 'assistant',
          content: res.answer,
          sources: res.docs,
          telemetry: res.telemetry,
          timestamp: new Date().toLocaleTimeString()
        });
      },
      error: (err) => {
        this.isChatLoading = false;
        this.messages.push({
          role: 'assistant',
          content: '⚠️ Error retrieving answer: ' + (err.error?.detail || err.message),
          timestamp: new Date().toLocaleTimeString()
        });
      }
    });
  }

  toggleSources(index: number): void {
    this.expandedSourceMessageIndex = this.expandedSourceMessageIndex === index ? null : index;
  }

  onAudioSelected(event: any): void {
    const file = event.target.files[0];
    if (!file) return;

    this.isTranscribing = true;
    this.api.transcribeAudio(file).subscribe({
      next: (res) => {
        this.isTranscribing = false;
        this.queryText = res.transcript;
      },
      error: (err) => {
        this.isTranscribing = false;
        alert('Voice transcription failed: ' + (err.error?.detail || err.message));
      }
    });
  }

  // TAB 2: Document Management
  loadDocuments(): void {
    this.api.getDocuments(this.activeCollection).subscribe({
      next: (docs) => {
        this.documents = docs;
      },
      error: (err) => console.error('Failed to load documents:', err)
    });
  }

  onFilesSelected(event: any): void {
    this.selectedFilesToUpload = Array.from(event.target.files);
  }

  uploadDocuments(): void {
    if (this.selectedFilesToUpload.length === 0) return;
    this.isUploading = true;
    this.uploadAlert = null;

    this.api.uploadDocuments(this.selectedFilesToUpload, this.activeCollection, this.preventDuplicates).subscribe({
      next: (res) => {
        this.isUploading = false;
        this.selectedFilesToUpload = [];
        const successCount = res.results.filter((r: any) => r.status === 'success').length;
        const dupCount = res.results.filter((r: any) => r.status === 'duplicate').length;

        if (successCount > 0 && dupCount === 0) {
          this.uploadAlert = { type: 'success', message: `Successfully indexed ${successCount} document(s)!` };
        } else if (dupCount > 0 && successCount === 0) {
          this.uploadAlert = { type: 'info', message: `Duplicate document(s) detected (${dupCount}). Document already indexed!` };
        } else {
          this.uploadAlert = { type: 'info', message: `Indexed ${successCount} document(s), skipped ${dupCount} duplicate(s).` };
        }
        this.loadDocuments();
      },
      error: (err) => {
        this.isUploading = false;
        this.uploadAlert = { type: 'error', message: 'Upload failed: ' + (err.error?.detail || err.message) };
      }
    });
  }

  deleteDocument(filename: string): void {
    if (!confirm(`Delete document "${filename}" and its vector chunks?`)) return;
    this.api.deleteDocument(filename, this.activeCollection).subscribe({
      next: () => {
        this.loadDocuments();
      },
      error: (err) => alert('Delete failed: ' + (err.error?.detail || err.message))
    });
  }

  onUpdateFileSelected(event: any): void {
    this.updateFile = event.target.files[0] || null;
  }

  updateDocument(): void {
    if (!this.updateTargetDoc || !this.updateFile) return;
    this.isUpdating = true;
    this.api.updateDocument(this.updateFile, this.updateTargetDoc, this.activeCollection).subscribe({
      next: (res) => {
        this.isUpdating = false;
        this.updateFile = null;
        alert(`Document "${this.updateTargetDoc}" replaced with ${res.chunks_added} updated chunks!`);
        this.loadDocuments();
      },
      error: (err) => {
        this.isUpdating = false;
        alert('Update failed: ' + (err.error?.detail || err.message));
      }
    });
  }

  // TAB 3: Advanced Retrieval Diagnostics
  runRetrievalDiagnostic(): void {
    if (!this.debugQuery.trim()) return;
    this.isDebugLoading = true;

    const filter = this.metadataFilterDoc !== 'ALL' ? { filename: this.metadataFilterDoc } : null;

    this.api.retrieveDebug({
      collection_name: this.activeCollection,
      query: this.debugQuery,
      search_type: this.searchType,
      k: this.retrievalK,
      fetch_k: this.retrievalFetchK,
      lambda_mult: this.retrievalLambda,
      score_threshold: this.searchType === 'similarity_score_threshold' ? this.retrievalScoreThreshold : null,
      metadata_filter: filter,
    }).subscribe({
      next: (res) => {
        this.isDebugLoading = false;
        this.debugResult = res;
        this.selectedInspectRank = 1;
      },
      error: (err) => {
        this.isDebugLoading = false;
        alert('Diagnostic failed: ' + (err.error?.detail || err.message));
      }
    });
  }

  getSelectedDebugItem(): any {
    if (!this.debugResult || !this.debugResult.items.length) return null;
    return this.debugResult.items.find(i => i.rank === this.selectedInspectRank) || this.debugResult.items[0];
  }

  // TAB 4: RAG Evaluation
  generateEvaluationDataset(): void {
    this.isGeneratingDataset = true;
    this.api.generateEvalDataset(this.activeCollection, this.numEvalQuestions).subscribe({
      next: (res) => {
        this.isGeneratingDataset = false;
        this.evalDataset = res.dataset;
      },
      error: (err) => {
        this.isGeneratingDataset = false;
        alert('Failed generating dataset: ' + (err.error?.detail || err.message));
      }
    });
  }

  runEvaluationBenchmark(): void {
    if (!this.evalDataset.length) return;
    this.isRunningEval = true;
    this.evalResponse = null;

    this.api.runEvaluation(this.activeCollection, this.evalDataset, this.evalEngine).subscribe({
      next: (res) => {
        this.isRunningEval = false;
        this.evalResponse = res;
      },
      error: (err) => {
        this.isRunningEval = false;
        alert('Evaluation run failed: ' + (err.error?.detail || err.message));
      }
    });
  }

  // TAB 5: ChromaDB Explorer
  switchTab(tab: 'chat' | 'docs' | 'retrieval' | 'eval' | 'explorer'): void {
    this.activeTab = tab;
    if (tab === 'explorer') {
      this.loadExplorerData();
    }
  }

  loadExplorerData(): void {
    this.isLoadingExplorer = true;
    this.api.getExplorerData(this.activeCollection).subscribe({
      next: (res) => {
        this.isLoadingExplorer = false;
        this.explorerData = res;
        if (res.chunks.length > 0) {
          this.selectedInspectChunkId = res.chunks[0].chunk_id;
        }
      },
      error: (err) => {
        this.isLoadingExplorer = false;
        console.error('Failed loading explorer data:', err);
      }
    });
  }

  getFilteredChunks(): any[] {
    if (!this.explorerData) return [];
    if (!this.chunkFilterKeyword.trim()) return this.explorerData.chunks;
    const kw = this.chunkFilterKeyword.toLowerCase();
    return this.explorerData.chunks.filter(
      c => c.content.toLowerCase().includes(kw) || c.filename.toLowerCase().includes(kw) || c.chunk_id.toLowerCase().includes(kw)
    );
  }

  getSelectedChunkContent(): any {
    if (!this.explorerData || !this.selectedInspectChunkId) return null;
    return this.explorerData.chunks.find(c => c.chunk_id === this.selectedInspectChunkId) || null;
  }

  // Helper for PCA SVG coordinates
  getPcaSvgX(point: PcaPoint): number {
    if (!this.explorerData || !this.explorerData.pca_points.length) return 200;
    const xs = this.explorerData.pca_points.map(p => p.x);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const range = (maxX - minX) || 1;
    return 40 + ((point.x - minX) / range) * 420;
  }

  getPcaSvgY(point: PcaPoint): number {
    if (!this.explorerData || !this.explorerData.pca_points.length) return 150;
    const ys = this.explorerData.pca_points.map(p => p.y);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const range = (maxY - minY) || 1;
    return 30 + ((point.y - minY) / range) * 240;
  }

  getUniquePcaDocuments(): string[] {
    if (!this.explorerData) return [];
    return Array.from(new Set(this.explorerData.pca_points.map(p => p.filename)));
  }

  getDocumentColor(filename: string): string {
    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#06b6d4', '#ef4444'];
    let hash = 0;
    for (let i = 0; i < filename.length; i++) {
      hash = filename.charCodeAt(i) + ((hash << 5) - hash);
    }
    const idx = Math.abs(hash) % colors.length;
    return colors[idx];
  }
}
