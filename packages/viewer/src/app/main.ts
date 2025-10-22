// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { mount } from "svelte";

import "../app.css";

import App from "./Entrypoint.svelte";

const app = mount(App, { target: document.getElementById("app")! });

export default app;
