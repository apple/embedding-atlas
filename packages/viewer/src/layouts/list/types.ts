export interface ListLayoutState {
  showTable?: boolean;
  showEmbedding?: boolean;
  showCharts?: boolean;

  chartsOrder?: string[];
  chartVisibility?: Record<string, boolean>;
}
