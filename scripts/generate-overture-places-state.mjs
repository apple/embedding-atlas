// Copyright (c) 2026. Licensed under MIT License.

import { deflateRawSync } from "node:zlib";

const baseUrl = (process.argv[2] ?? "http://localhost:5055").replace(/\/$/, "");

const liquid = (template) => ({
  display: "full",
  renderer: "liquid-template",
  options: { template: template.trim() },
});

const state = {
  version: "0.24.0",
  charts: {
    1: {
      type: "embedding",
      title: "Places",
      data: {
        x: "lon",
        y: "lat",
        isGis: true,
        viewportHint: {
          centerX: -0.0002186359097180457,
          centerY: -0.7093139790175016,
          rangeX: 359.9956337367595,
          rangeY: 168.561572402435,
          rowCount: 73631092,
        },
      },
    },
    2: { type: "predicates", title: "Filter places" },
    3: { type: "instances", title: "Place details" },
  },
  chartStates: {
    1: {
      viewport: {
        x: -0.0002186359097180457,
        y: -0.7093139790175016,
        scale: 0.005277841790129437,
      },
    },
  },
  layouts: {
    1: { type: "list", name: "Overture Places", chartIds: ["1", "2", "3"] },
  },
  layoutOrder: ["1"],
  currentLayout: "1",
  columnStyles: {
    id: { display: "hidden" },
    geometry: { display: "hidden" },
    categories: { display: "hidden" },
    confidence: liquid(`
      {% if value != nil %}
        {% assign pct = value | times: 100 %}
        <div style="display:flex;align-items:center;gap:8px;min-width:180px">
          <div style="height:7px;flex:1;background:#e2e8f0;border-radius:999px;overflow:hidden">
            <div style="height:100%;width:{{ pct }}%;background:#2563eb;border-radius:999px"></div>
          </div>
          <strong style="font-variant-numeric:tabular-nums">{{ pct | round: 1 }}%</strong>
        </div>
      {% else %}
        <span style="opacity:.55">Not scored</span>
      {% endif %}
    `),
    websites: liquid(`
      {% if value %}
        <div style="display:flex;flex-direction:column;gap:2px">
          {% for url in value limit: 3 %}
            <a href="{{ url | escape }}" style="color:#2563eb;text-decoration:underline">
              {{ url | remove: "https://" | remove: "http://" | remove: "www." | truncate: 52 | escape }}
            </a>
          {% endfor %}
          {% if value.size > 3 %}<small style="opacity:.6">+{{ value.size | minus: 3 }} more</small>{% endif %}
        </div>
      {% else %}<span style="opacity:.45">—</span>{% endif %}
    `),
    emails: liquid(`
      {% if value %}
        <div style="display:flex;flex-wrap:wrap;gap:4px 10px">
          {% for email in value limit: 3 %}
            <a href="mailto:{{ email | escape }}" style="color:#2563eb;text-decoration:underline">{{ email | escape }}</a>
          {% endfor %}
          {% if value.size > 3 %}<small style="opacity:.6">+{{ value.size | minus: 3 }} more</small>{% endif %}
        </div>
      {% else %}<span style="opacity:.45">—</span>{% endif %}
    `),
    socials: liquid(`
      {% if value %}
        <div style="display:flex;flex-direction:column;gap:2px">
          {% for url in value limit: 3 %}
            <a href="{{ url | escape }}" style="color:#2563eb;text-decoration:underline">
              {{ url | remove: "https://" | remove: "http://" | remove: "www." | truncate: 52 | escape }}
            </a>
          {% endfor %}
          {% if value.size > 3 %}<small style="opacity:.6">+{{ value.size | minus: 3 }} more</small>{% endif %}
        </div>
      {% else %}<span style="opacity:.45">—</span>{% endif %}
    `),
    phones: liquid(`
      {% if value %}
        <div style="display:flex;flex-wrap:wrap;gap:4px 10px">
          {% for phone in value limit: 3 %}
            <a href="tel:{{ phone | escape }}" style="color:#2563eb;text-decoration:underline">{{ phone | escape }}</a>
          {% endfor %}
          {% if value.size > 3 %}<small style="opacity:.6">+{{ value.size | minus: 3 }} more</small>{% endif %}
        </div>
      {% else %}<span style="opacity:.45">—</span>{% endif %}
    `),
    brand: liquid(`
      {% if value.names.primary or value.wikidata %}
        <div style="display:flex;align-items:center;gap:8px">
          {% if value.names.primary %}<strong>{{ value.names.primary | escape }}</strong>{% endif %}
          {% if value.wikidata %}
            <a href="https://www.wikidata.org/wiki/{{ value.wikidata | escape }}" style="font-size:.8em;color:#2563eb;text-decoration:underline">
              {{ value.wikidata | escape }}
            </a>
          {% endif %}
        </div>
      {% else %}<span style="opacity:.45">Independent / unbranded</span>{% endif %}
    `),
    addresses: liquid(`
      {% if value %}
        <div style="display:flex;flex-direction:column;gap:5px">
          {% for address in value limit: 2 %}
            <div>
              <span aria-hidden="true">📍</span>
              {% if address.freeform %}{{ address.freeform | escape }}{% endif %}
              {% if address.locality %}{% if address.freeform %}, {% endif %}{{ address.locality | escape }}{% endif %}
              {% if address.region %}, {{ address.region | escape }}{% endif %}
              {% if address.postcode %} {{ address.postcode | escape }}{% endif %}
              {% if address.country %}<small style="opacity:.65"> · {{ address.country | escape }}</small>{% endif %}
            </div>
          {% endfor %}
          {% if value.size > 2 %}<small style="opacity:.6">+{{ value.size | minus: 2 }} more addresses</small>{% endif %}
        </div>
      {% else %}<span style="opacity:.45">No address</span>{% endif %}
    `),
    names: liquid(`
      <div style="line-height:1.35">
        <div style="font-size:1.12em;font-weight:700">{{ value.primary | default: "Unnamed place" | escape }}</div>
        {% if value.rules %}
          <div style="display:flex;flex-direction:column;gap:2px;margin-top:4px">
            {% for rule in value.rules limit: 3 %}
              {% if rule.value %}
                <div style="font-size:.84em;opacity:.72">
                  {{ rule.variant | default: "alternative" | replace: "_", " " | capitalize | escape }}{% if rule.language %} · {{ rule.language | upcase | escape }}{% endif %}:
                  {{ rule.value | escape }}
                </div>
              {% endif %}
            {% endfor %}
          </div>
        {% endif %}
      </div>
    `),
    sources: liquid(`
      {% if value %}
        <details style="font-size:.82em">
          <summary style="cursor:pointer">{{ value.size }} provenance record{% if value.size != 1 %}s{% endif %}</summary>
          <div style="display:flex;flex-direction:column;gap:5px;margin-top:5px">
            {% for source in value limit: 5 %}
              <div>
                <strong>{{ source.provider | default: source.dataset | default: "unknown" | escape }}</strong>
                {% if source.resource %} · {{ source.resource | escape }}{% endif %}
                {% if source.update_time %}<small style="opacity:.65"> · {{ source.update_time | truncate: 10, "" | escape }}</small>{% endif %}
              </div>
            {% endfor %}
          </div>
        </details>
      {% else %}<span style="opacity:.45">No provenance</span>{% endif %}
    `),
    taxonomy: liquid(`
      <div style="line-height:1.4">
        <strong>{{ value.primary | replace: "_", " " | capitalize | escape }}</strong>
        {% if value.hierarchy %}
          <div style="font-size:.84em;opacity:.68">
            {% for item in value.hierarchy %}
              {% unless forloop.first %} › {% endunless %}{{ item | replace: "_", " " | escape }}
            {% endfor %}
          </div>
        {% endif %}
        {% if value.alternates %}
          <div style="font-size:.8em;margin-top:3px">Also: {{ value.alternates | join: ", " | replace: "_", " " | escape }}</div>
        {% endif %}
      </div>
    `),
    version: { display: "hidden" },
    bbox: { display: "hidden" },
    lon: liquid(`<span style="font-variant-numeric:tabular-nums">{{ value | round: 5 }}°</span>`),
    lat: liquid(`<span style="font-variant-numeric:tabular-nums">{{ value | round: 5 }}°</span>`),
    operating_status: liquid(`
      {% if value %}
        <span style="display:inline-block;padding:2px 8px;border-radius:999px;background:#dcfce7;color:#166534;font-weight:600">
          {{ value | replace: "_", " " | capitalize | escape }}
        </span>
      {% else %}<span style="opacity:.55">Unspecified</span>{% endif %}
    `),
    basic_category: liquid(`
      {% if value %}
        <span style="display:inline-block;padding:2px 8px;border-radius:999px;background:#dbeafe;color:#1e40af;font-weight:600">
          {{ value | replace: "_", " " | capitalize | escape }}
        </span>
      {% else %}<span style="opacity:.45">—</span>{% endif %}
    `),
  },
};

const payload = deflateRawSync(Buffer.from(JSON.stringify(state))).toString("base64url");
console.log(`${baseUrl}/#?state=${payload}`);
