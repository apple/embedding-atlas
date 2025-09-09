// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { createElement, useEffect, useRef } from "react";

import {
  EmbeddingView as EmbeddingViewComponent,
  EmbeddingViewMosaic as EmbeddingViewMosaicComponent,
  type EmbeddingViewMosaicProps,
  type EmbeddingViewProps,
} from "./component.js";
import { Table as TableComponent, type TableProps } from "./table.js";
import { EmbeddingAtlas as EmbeddingAtlasComponent, type EmbeddingAtlasProps } from "./viewer.js";

function makeReactWrapper<Props>(
  ComponentClass: any,
  tag: string = "div",
  style: any = { display: "flex" },
): (props: Props) => any {
  return (props) => {
    const container = useRef<any>(null);
    const component = useRef<any>(null);
    useEffect(() => {
      let c = new ComponentClass(container.current!, props);
      component.current = c;
      return () => {
        component.current?.destroy();
      };
    }, []);
    useEffect(() => {
      component.current?.update(props);
    }, [props]);
    return createElement(tag, { ref: container, style: style });
  };
}

// Create wrappers
const EmbeddingAtlas = makeReactWrapper<EmbeddingAtlasProps>(EmbeddingAtlasComponent, "div", {
  display: "flex",
  width: "100%",
  height: "100%",
});

const EmbeddingView = makeReactWrapper<EmbeddingViewProps>(EmbeddingViewComponent);
const EmbeddingViewMosaic = makeReactWrapper<EmbeddingViewMosaicProps>(EmbeddingViewMosaicComponent);
const Table = makeReactWrapper<TableProps>(TableComponent);

export * from "./index.js";
export { EmbeddingAtlas, EmbeddingView, EmbeddingViewMosaic, Table };
