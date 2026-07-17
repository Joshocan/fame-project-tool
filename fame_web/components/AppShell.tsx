"use client";

import Link from "next/link";
import type { Route } from "next";
import { usePathname } from "next/navigation";
import { PropsWithChildren } from "react";
import { ContainerRecord } from "@/lib/models";

type IntegrationConfig = {
  githubTokenConfigured: boolean;
  anthropicKeyConfigured: boolean;
  chromaPath: string;
  neo4jUri: string;
};

type AppShellProps = PropsWithChildren<{
  config: IntegrationConfig;
  containers: ContainerRecord[];
}>;

const navItems: { href: Route; label: string; exact: boolean }[] = [
  { href: "/", label: "Overview", exact: true },
  { href: "/containers", label: "Systems", exact: false },
  { href: "/runs/new", label: "New Run", exact: false }
];

const integrations = (cfg: IntegrationConfig) => [
  { label: "Anthropic key", ok: cfg.anthropicKeyConfigured },
  { label: "ChromaDB", ok: !!cfg.chromaPath },
  { label: "Neo4j", ok: !!cfg.neo4jUri }
];

function PlusIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  );
}

export function AppShell({ children, config, containers }: AppShellProps) {
  const pathname = usePathname();

  return (
    <div className="shell">
      <header className="topBar">
        <div className="topBarBrand">
          <span className="topBarEyebrow">FAME</span>
          <span className="topBarTitle">Web</span>
        </div>
        <div className="topBarDivider" />
        <nav className="topBarNav">
          {navItems.map(({ href, label, exact }) => {
            const active = exact ? pathname === href : pathname.startsWith(href);
            return (
              <Link key={href} href={href} className={`topNavLink${active ? " active" : ""}`}>
                {label}
              </Link>
            );
          })}
        </nav>
      </header>

      <div className="contentArea">
        <aside className="sourcesPanel">
          <div className="panelHeader">
            <span className="panelTitle">Systems</span>
            <Link href="/containers" className="panelIconBtn" title="New system">
              <PlusIcon />
            </Link>
          </div>
          <div className="panelBody stack sourcesListWrap">
            <Link href="/containers" className="button ghost">
              New system
            </Link>
            {containers.length ? (
              <div className="stack sourceList">
                {containers.map((c) => (
                  <Link key={c.id} href={`/containers/${c.id}` as Route} className="sourceCard">
                    <strong>{c.name}</strong>
                    <span className="muted">
                      {c.files.length} file{c.files.length !== 1 ? "s" : ""}
                    </span>
                  </Link>
                ))}
              </div>
            ) : (
              <p className="muted">No systems yet.</p>
            )}
          </div>
        </aside>

        <main className="mainPanel">{children}</main>

        <aside className="studioPanel">
          <div className="panelHeader">
            <span className="panelTitle">Studio</span>
          </div>
          <div className="panelBody">
            <div className="studioSection">
              <p className="studioSectionTitle">Integrations</p>
              <div className="stack" style={{ gap: "0.45rem" }}>
                {integrations(config).map(({ label, ok }) => (
                  <div key={label} className="statusRow">
                    <span className={`statusDot ${ok ? "ok" : "off"}`} />
                    <span>{label}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="studioSection">
              <p className="studioSectionTitle">Quick actions</p>
              <div className="studioActions">
                <Link href="/containers" className="button ghost">
                  Browse systems
                </Link>
                <Link href="/runs/new" className="button ghost">
                  Configure a run
                </Link>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
