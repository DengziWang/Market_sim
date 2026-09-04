# Wie misst man, ob ein (Krypto-)Markt gesund ist?

### Wissenschaftlicher Bericht mit Anbindung an die Simulationsauswertung

> Vom Nutzer bereitgestellt am 2026-09-04 (als Antworttext, nicht als Datei) und hier unverändert
> inhaltlich als Referenzdokument für `market_health/` abgelegt.

## 1. Einordnung

Es gibt keine einzelne Kennzahl für "Marktgesundheit". Die Marktmikrostruktur-Literatur
(Übersichtsarbeiten wie O'Hara, sowie neuere bibliometrische Reviews) gliedert Marktqualität
konsistent in mehrere Dimensionen: **Liquidität, Volatilität, Preiseffizienz/Informationsgehalt,
Fragilität/Toxizität** sowie — speziell im Krypto-Kontext — **Authentizität des Volumens** und
**Konzentration/Fairness**. Eine häufig zitierte Studie zu sub-rationalen/algorithmischen Agenten
(Boehmer, Fong & Wu-Rahmen, angewendet u. a. in aktuellen Agent-Modellierungsarbeiten) bewertet
Marktqualität explizit entlang dreier Kerndimensionen: Liquidität, Volatilität und
Informationseffizienz. Für Kryptomärkte muss dies um Wash-Trading-Erkennung und
Fragilitäts-/Crash-Analyse erweitert werden, da diese Phänomene dort empirisch dominant sind.

## 2. Dimension 1: Liquidität

**Bid-Ask-Spread / effektiver relativer Spread.** Der relative effektive Spread wird berechnet als
RES = (p(t) − m(t)) / m(t), wobei p(t) der Ausführungspreis und m(t) der Mid-Price zum
Ausführungszeitpunkt ist. Je enger der Spread, desto liquider der Markt.

**Orderbuch-Tiefe (Depth).** Verfügbares Volumen auf den ersten N Preisstufen je Seite. Eine
aktuelle Analyse (CME Group, 2025) zeigt, dass reine Tiefe-Betrachtung irreführend sein kann und
Preis-Impact-Modelle (z. B. Square-Root-Impact-Modelle) die tatsächliche Ausführungsliquidität
besser abbilden als die reine Orderbuchtiefe allein.

**Amihud-Illiquiditätsmaß.** Verhältnis von absoluter Rendite zu gehandeltem Volumen in einem
Zeitfenster; eines der am häufigsten verwendeten Liquiditätsmaße in der Finanzliteratur (Amihud
2002), da es mit niedrig-frequenten Daten auskommt.

**Kyle's Lambda (Preis-Impact-Koeffizient).** Misst, wie stark der Preis auf eine Einheit
Netto-Orderfluss reagiert — ein direktes Maß für die Fähigkeit des Marktes, Kapital ohne
signifikante Preisbewegung aufzunehmen.

**Resilienz.** Wie schnell kehrt Orderbuchtiefe/Spread nach einem Schock zum
Ausgangsniveau zurück? Dies ist eine dynamische, prozessuale Größe (siehe BIS-Arbeit von
Muranaga & Shimizu), die statische Momentaufnahmen ergänzt und für Stresstests besonders relevant
ist.

**Anbindung an Simulation:** berechenbar direkt aus `orderbook_snapshots.parquet` (Spread, Tiefe)
und `trades.parquet` (Amihud, Kyle's Lambda via Regression von Preisänderung auf signierten
Orderfluss).

## 3. Dimension 2: Volatilität

**Realisierte Volatilität** über verschiedene Zeitfenster, sowie **Volatilitäts-Clustering**
(Autokorrelation der quadrierten/absoluten Returns) — eine der robustesten "stilisierten
Fakten" empirischer Finanzmarktdaten, gegen die jedes Agentenmodell validiert werden sollte.

**Sprünge (Jumps).** Anteil der Preisbewegung, der auf diskrete Sprünge statt kontinuierliche
Diffusion zurückgeht; relevant zur Unterscheidung "normale" vs. "toxische/manipulierte"
Preisbildung.

**Anbindung an Simulation:** aus `orderbook_snapshots.parquet`/Mid-Price-Zeitreihe pro Asset
berechenbar, inklusive Autokorrelationsfunktion der Returns zur Prüfung des Clustering-Effekts.

## 4. Dimension 3: Preiseffizienz / Informationsgehalt

**Variance Ratio (Lo & MacKinlay 1988; Smith 1994).** Prüft, ob sich Preise wie ein Random Walk
verhalten (Effizienzmarkthypothese); Abweichungen von 1 deuten auf Ineffizienz
(Mean-Reversion oder Momentum-Verzerrung) hin. Für Rohstoffmärkte wurde gezeigt, dass die
Messung von Preiseffizienz hochfrequente Daten benötigt — bei Tagesdaten sind die Proxies kaum
belastbar; 5-Minuten-Daten liefern eine moderat verlässliche Annäherung. Das ist direkt auf die
Simulation übertragbar: die Analyse sollte auf möglichst granularen Tick-/Snapshot-Daten laufen.

**Hasbrouck Pricing Error / Information Share.** Zerlegt beobachtete Preise in einen
"effizienten" (fundamentalen) und einen Microstructure-Noise-Anteil; hohe Pricing-Error-Varianz
deutet auf geringe Informationseffizienz hin.

**Autokorrelation der Returns.** Einfacher, robuster Zusatzindikator für Ineffizienz
(positive Autokorrelation → Momentum/Trend-Verzerrung, negative → übermäßige Mean-Reversion,
z. B. durch Market-Maker-Bouncing).

**Anbindung an Simulation:** Da die Simulation den "wahren" Fundamentalwert (sofern modelliert,
z. B. über die Fundamentalisten-/Fair-Value-Bots) kennt, kann zusätzlich die **Abweichung des
Marktpreises vom bekannten Fundamentalwert** direkt gemessen werden — ein Vorteil gegenüber
realen Märkten, wo der Fundamentalwert unbekannt ist.

## 5. Dimension 4: Fragilität, Toxizität und Krisenanfälligkeit

**VPIN (Volume-Synchronized Probability of Informed Trading).** Ein aus der Marktmikrostruktur-
Literatur stammender Frühwarnindikator für toxischen Orderfluss, der in Zusammenhang mit dem
Flash Crash vom 6. Mai 2010 diskutiert wird. Hohe VPIN-Werte deuten auf ein Ungleichgewicht
zwischen informierten und uninformierten Marktteilnehmern hin, das Market-Maker zum Rückzug
zwingen kann.

**Order-Flow-Imbalance.** Netto-Kauf- vs. Verkaufsdruck im Zeitfenster; abrupte Imbalances sind
ein klassischer Crash-Vorläufer.

**Cancel-to-Trade-Ratio.** Hohe Werte deuten auf Quote-Stuffing/Spoofing-artiges Verhalten hin
(relevant, da manche Bot-Typen wie MEV- oder Latency-Arbitrage-Bots exzessiv Orders
einstellen/stornieren können).

**Agent-basierte Crash-Forschung als Referenzrahmen.** Die einflussreichste Arbeit zu diesem
Thema ist die kalibrierte Rekonstruktion des Flash Crash vom 6. Mai 2010 (Paddrik et al. 2012,
aufbauend auf der empirischen Analyse von Kirilenko et al.), die zeigt, wie das Zusammenspiel
aus institutionellen Verkaufsalgorithmen, High-Frequency-Tradern und Market-Maker-
Bestandsgrenzen einen Flash Crash erzeugen und wieder auflösen kann. Neuere Arbeiten (Gao et al.
2022, JASSS) erweitern dies zu einem hochfrequenten Simulator, der gezielt "Spiking Trader"
einführt, um Mini-Flash-Crashes zu erzeugen, und finden, dass insbesondere das
**Bestandslimit der Market-Maker** (Inventory Limit) die Häufigkeit und Amplitude solcher
Ereignisse signifikant beeinflusst — ein direkt übertragbarer, testbarer Parameter für eure
Simulation.

**Marktökologie-Perspektive.** Besonders relevant für eure Forschungsfrage ist die Arbeit von
Scholl, Calinescu & Farmer (2021, PNAS), die Marktversagen als Konsequenz der **Zusammensetzung
und Diversität der Strategie-Population** erklärt, nicht nur einzelner Agentenfehler. Das stützt
methodisch genau euren Ansatz, verschiedene Bot-Kompositionen systematisch zu vergleichen: eine
geringe Diversität (z. B. sehr viele strukturell ähnliche Bots) kann die Fragilität erhöhen, auch
wenn jeder einzelne Bot "rational" handelt.

**Anbindung an Simulation:** VPIN und Order-Flow-Imbalance aus `trades.parquet`/
`order_events.parquet`; Cancel-to-Trade-Ratio direkt aus `order_events.parquet`; Crash-Erkennung
als Post-Processing-Routine auf der Mid-Price-Zeitreihe (Schwellenwert-basierte Erkennung von
Tiefe/Dauer/Erholungszeit signifikanter Preiseinbrüche, analog Paddrik et al.).

**Liquidations-Kaskaden — Modellierungsentscheidung:** Für die Simulation wurde bewusst ein
vereinfachtes Margin-Modell festgelegt (Hebel-Position, Maintenance-Margin-Puffer, ein
EMA-basierter "Mark Price" statt eines echten externen Index, sowie partielle statt ausschließlich
vollständige Liquidation), da diese Elemente die kurzfristige Kaskaden-*Preisdynamik* prägen —
während Funding-Rate-Zahlungen, Insurance Fund und Auto-Deleveraging primär die langfristige
Verteilung von Gewinnen/Verlusten zwischen Marktteilnehmern betreffen und für die
Kernfragestellung (Marktgesundheit unter Bot-Dominanz) nachrangig sind. Diese drei Mechanismen
sind als optionale Phase-2-Erweiterung vorgesehen, falls die Perpetual-Spot-Kopplung selbst
Untersuchungsgegenstand werden soll.

## 6. Dimension 5: Authentizität des Volumens (Wash-Trading-Erkennung)

Die akademisch etablierte Methode (Cong, Li, Tang & Yang, publiziert in *Management Science*)
nutzt drei statistische Signaturen, die authentischen Handel von fabriziertem Handel
unterscheiden:

1. **Erste-signifikante-Ziffer-Verteilung (Benford's Law):** authentische Handelsgrößen folgen
   Benford's Law; fabrizierte Trades weichen systematisch ab.
2. **Rundungs-/Clustering-Analyse:** künstlich erzeugte Trades häufen sich auffällig oft bei
   runden Zahlen.
3. **Tail-Verteilung der Handelsgrößen:** authentische Größen folgen typischerweise einem
   Pareto/Power-Law-Muster im Tail; Abweichungen deuten auf Manipulation hin.

Zusätzlich, komplementär für On-Chain-/Wallet-Daten: Erkennung von **Wallet-/Adress-Ringen**, bei
denen Kapital zyklisch zwischen einer kleinen Gruppe von Konten hin- und herbewegt wird
(Methodik u. a. aus einer aktuellen Columbia-Studie zu Polymarket, die Cluster von mehreren
zehntausend Wallets mit wiederholtem Öffnen/Schließen von Positionen unter sich selbst
identifizierte).

**Anbindung an Simulation:** Da eure Simulation die "wahre" Agenten-Identität kennt (im
Gegensatz zu realen, pseudonymen Märkten), lässt sich Wash-Trading **doppelt validieren**: einmal
über die statistischen Proxy-Tests (wie in der Realität üblich), einmal über die tatsächliche
Ground-Truth (welcher Anteil des Volumens stammt nachweislich von als "Wash-Trading-Bot"
konfigurierten Agenten). Das erlaubt, die Erkennungsgüte der statistischen Methode selbst zu
validieren — ein zusätzlicher, methodisch wertvoller Output eurer Simulation.

## 7. Dimension 6: Konzentration/Fairness

**Herfindahl-Hirschman-Index (HHI)** über den Volumenanteil pro Agent bzw. Agent-Typ. Hohe
Konzentration (wie empirisch bei MEV-Sandwich-Bots beobachtet, wo ein einzelner Bot einen
Großteil der Angriffe eines Marktsegments verantwortet) deutet auf strukturelle Fragilität und
mangelnde Wettbewerbsfähigkeit hin — ein "gesunder" Markt sollte eine breitere Beteiligung
aufweisen.

## 8. Zusammengesetzter Marktgesundheits-Score

Es existiert keine einzelne, universell akzeptierte Formel für einen Gesamt-Score — die
Literatur behandelt die Dimensionen bewusst getrennt, da sie unterschiedliche, teils
gegenläufige Aspekte erfassen (z. B. kann hohe Liquidität mit hoher Toxizität einhergehen).

Empfehlung für die Simulation:

1. Jede Metrik pro Lauf berechnen und **z-normalisieren relativ zur gesamten Experiment-Batch**
   (damit Werte über verschiedene Bot-Kompositionen vergleichbar werden).
2. Vorzeichen so ausrichten, dass "höher = gesünder" (z. B. Spread und VPIN invertieren).
3. Score als **gewichtete Summe** über die sechs Dimensionen bilden, mit konfigurierbaren
   Gewichten (Default: Gleichgewichtung), damit im Bericht transparent bleibt, welche
   Werturteile in den Score einfließen.
4. Den Gesamt-Score **immer zusammen mit den Einzeldimensionen** berichten, nie isoliert —
   um genau die Fehlinterpretation zu vermeiden, vor der die Fake-Volumen-Literatur ausdrücklich
   warnt (aggregierte Kennzahlen über heterogene Börsen/Bedingungen können in die Irre führen).

## 9. Kurz-Referenzliste (Auswahl)

- Amihud, Y. (2002). *Illiquidity and Stock Returns.* Journal of Financial Markets.
- Cong, L. W., Li, X., Tang, K., & Yang, Y. *Crypto Wash Trading.* Management Science / NBER.
- Farmer, J. D., Patelli, P., & Zovko, I. I. (2005). *The Predictive Power of Zero Intelligence
  in Financial Markets.* PNAS.
- Gao, K., Vytelingum, P., Weston, S., Luk, W., & Guo, C. (2022/2024). *High-Frequency Financial
  Market Simulation and Flash Crash Scenarios Analysis: An Agent-Based Modelling Approach.*
  Journal of Artificial Societies and Social Simulation (JASSS).
- Gode, D. K., & Sunder, S. (1993). *Allocative Efficiency of Markets with Zero-Intelligence
  Traders.* Journal of Political Economy.
- Kirilenko, A., Kyle, A. S., Samadi, M., & Tuzun, T. (2017). *The Flash Crash: High-Frequency
  Trading in an Electronic Market.* Journal of Finance.
- Paddrik, M., Hayes, R., Todd, A., Yang, S., Beling, P., & Scherer, W. (2012). *An Agent Based
  Model of the E-Mini S&P 500 Applied to Flash Crash Analysis.* IEEE CIFEr.
- Scholl, M. P., Calinescu, A., & Farmer, J. D. (2021). *How Market Ecology Explains Market
  Malfunction.* PNAS.
- Muranaga, J., & Shimizu, T. *Market Microstructure and Market Liquidity.* BIS.

## 10. Direkter Anschluss an die Simulation

Alle in diesem Bericht genannten Metriken sind so gewählt, dass sie **ausschließlich aus den in
Abschnitt 7 des Claude-Code-Prompts spezifizierten Export-Dateien** (`trades.parquet`,
`orderbook_snapshots.parquet`, `order_events.parquet`, `agent_state.parquet`,
`run_metadata.json`) berechnet werden können, ohne die Simulation erneut ausführen zu müssen.
Das Analyse-Modul `market_health/` (siehe Abschnitt 9 des Prompts) soll pro Metrik eine eigene,
unabhängig testbare Funktion bereitstellen sowie eine Aggregations-Funktion für den
zusammengesetzten Score.

**Methodische Mindestanforderung:** Statistische Tests wie der Benford's-Law-Test zur
Wash-Trading-Erkennung benötigen eine ausreichende Stichprobengröße, um aussagekräftig zu sein.
Der Prompt schreibt deshalb eine Mindest-Trade-Anzahl von ca. 10.000–50.000 Trades pro
Simulationslauf sowie 20–50 unabhängige Seeds pro untersuchter Bot-Komposition vor, damit die
Auswertung Konfidenzintervalle statt Einzelpunkt-Schätzungen pro Metrik berichten kann.
