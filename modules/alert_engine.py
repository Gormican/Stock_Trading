"""
alert_engine.py — Email alerts for GPA signals and daily summaries
Sends to sgormican@gmail.com via Gmail SMTP App Password
"""
import smtplib
import logging
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger("AlertEngine")


def _html_header():
    return """
    <style>
      body { font-family: Arial, sans-serif; background:#f5f5f5; }
      .card { background:white; border-radius:8px; padding:20px; margin:10px 0;
              box-shadow:0 1px 4px rgba(0,0,0,0.1); }
      .gpa-high  { color:#16a34a; font-size:2em; font-weight:bold; }
      .gpa-mid   { color:#ca8a04; font-size:2em; font-weight:bold; }
      .gpa-low   { color:#dc2626; font-size:2em; font-weight:bold; }
      table { border-collapse:collapse; width:100%; }
      th { background:#1e3a5f; color:white; padding:8px; text-align:left; }
      td { padding:6px 8px; border-bottom:1px solid #eee; }
      tr:hover { background:#f9f9f9; }
      .pill { display:inline-block; padding:2px 8px; border-radius:12px;
              font-size:0.8em; font-weight:bold; }
      .buy  { background:#dcfce7; color:#166534; }
      .watch{ background:#fef9c3; color:#854d0e; }
      .sell { background:#fee2e2; color:#991b1b; }
    </style>
    """


class AlertEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.to  = cfg["alerts"]["email_to"]

    def _send(self, subject: str, html_body: str) -> bool:
        smtp_user = self.cfg["alerts"]["smtp_user"]
        smtp_pass = self.cfg["alerts"]["smtp_password"]
        if "YOUR_" in smtp_user or "YOUR_" in smtp_pass:
            log.warning("Email credentials not configured — skipping send")
            log.info(f"[ALERT PREVIEW] {subject}")
            return False
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = smtp_user
            msg["To"]      = self.to
            msg.attach(MIMEText(html_body, "html"))
            with smtplib.SMTP(self.cfg["alerts"]["smtp_host"],
                              self.cfg["alerts"]["smtp_port"]) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, self.to, msg.as_string())
            log.info(f"Email sent: {subject}")
            return True
        except Exception as e:
            log.error(f"Email send failed: {e}")
            return False

    # ── GPA Alert (triggered when stock hits >= 3.5) ──────────────────────────

    def send_gpa_alert(self, gpa_result: dict, snapshot: dict):
        symbol = gpa_result["symbol"]
        gpa    = gpa_result["gpa"]
        grade  = gpa_result["grade"]
        action = "BUY" if gpa_result.get("buy_signal") else "WATCH"
        price  = snapshot.get("price", 0)
        chg    = snapshot.get("change_pct", 0)

        color_class = "gpa-high" if gpa >= 3.5 else ("gpa-mid" if gpa >= 2.5 else "gpa-low")
        pill_class  = "buy" if action == "BUY" else "watch"

        pillars_rows = ""
        for pillar, data in gpa_result.get("pillars", {}).items():
            s   = data.get("score", 0)
            bar = int((s / 4.0) * 20)
            pillars_rows += f"""
            <tr>
              <td>{pillar.replace('_',' ').title()}</td>
              <td><span class="{color_class}" style="font-size:1em">{s:.2f}</span></td>
              <td>{'|' * bar}{'.' * (20 - bar)}</td>
            </tr>"""

        headlines = gpa_result.get("pillars", {}).get("sentiment", {}).get("detail", {})
        top_headlines = gpa_result.get("top_headlines", [])
        headline_html = "".join(
            f"<li>{h.get('title','')}</li>" for h in top_headlines[:3]
        )

        html = f"""
        {_html_header()}
        <div class="card">
          <h2>{symbol} — GPA Alert <span class="pill {pill_class}">{action}</span></h2>
          <p>
            <span class="{color_class}">{gpa:.2f}</span>
            &nbsp; Grade: <b>{grade}</b>
            &nbsp;|&nbsp; Price: <b>${price:.2f}</b>
            &nbsp;|&nbsp; Day: <b style="color:{'#16a34a' if chg>=0 else '#dc2626'}">{chg:+.2f}%</b>
          </p>
          <h3>GPA Breakdown</h3>
          <table>
            <tr><th>Pillar</th><th>Score (/ 4.0)</th><th>Visual</th></tr>
            {pillars_rows}
          </table>
          <h3>Key Drivers</h3>
          <p><b>Strengths:</b> {', '.join(gpa_result.get('top_drivers',[]))}</p>
          <p><b>Risks:</b> {', '.join(gpa_result.get('detractors',[])) or 'None identified'}</p>
          <h3>Top Headlines</h3>
          <ul>{headline_html}</ul>
          <hr>
          <p style="color:gray;font-size:0.8em">
            Taylor Trading Agent &mdash; {datetime.now().strftime('%Y-%m-%d %H:%M ET')} &mdash; Paper Mode
          </p>
        </div>"""

        return self._send(
            subject=f"[GPA {gpa:.2f}] {symbol} — {action} Signal | Taylor Trading",
            html_body=html,
        )

    # ── Daily Summary Email ───────────────────────────────────────────────────

    def send_daily_summary(self, all_scores: list, trades: list,
                           portfolio_value: float, daily_pnl: float):
        date_str = datetime.now().strftime("%A, %B %d, %Y")
        pnl_color = "#16a34a" if daily_pnl >= 0 else "#dc2626"

        # All-scores table
        rows = ""
        for r in sorted(all_scores, key=lambda x: x["gpa"], reverse=True):
            gpa     = r["gpa"]
            action  = "BUY" if r.get("buy_signal") else ("SELL" if r.get("sell_signal") else "HOLD")
            pill_cl = "buy" if action=="BUY" else ("sell" if action=="SELL" else "watch")
            gpa_cl  = "gpa-high" if gpa>=3.5 else ("gpa-mid" if gpa>=2.5 else "gpa-low")
            rows += f"""
            <tr>
              <td><b>{r['symbol']}</b></td>
              <td><span class="{gpa_cl}">{gpa:.2f}</span> {r['grade']}</td>
              <td><span class="pill {pill_cl}">{action}</span></td>
              <td>{', '.join(r.get('top_drivers', []))}</td>
            </tr>"""

        # Trades table
        trade_rows = ""
        for t in trades:
            trade_rows += f"""
            <tr>
              <td>{t.get('action')}</td><td>{t.get('symbol')}</td>
              <td>{t.get('shares')}</td><td>${t.get('price',0):.2f}</td>
              <td>${t.get('value',0):,.2f}</td><td>GPA {t.get('gpa',0):.2f}</td>
            </tr>"""

        html = f"""
        {_html_header()}
        <div class="card">
          <h1>Daily Trading Summary</h1>
          <h3>{date_str}</h3>
          <p>
            Portfolio: <b>${portfolio_value:,.2f}</b> &nbsp;|&nbsp;
            Day P&amp;L: <b style="color:{pnl_color}">${daily_pnl:+,.2f}</b>
          </p>
        </div>
        <div class="card">
          <h2>Stocks Evaluated Today</h2>
          <table>
            <tr><th>Symbol</th><th>GPA</th><th>Signal</th><th>Key Drivers</th></tr>
            {rows or '<tr><td colspan=4>No stocks evaluated today</td></tr>'}
          </table>
        </div>
        <div class="card">
          <h2>Trades Executed Today</h2>
          <table>
            <tr><th>Action</th><th>Symbol</th><th>Shares</th>
                <th>Price</th><th>Value</th><th>GPA</th></tr>
            {trade_rows or '<tr><td colspan=6>No trades today</td></tr>'}
          </table>
        </div>
        <p style="color:gray;font-size:0.8em">
          Taylor Trading Agent &mdash; Paper Mode &mdash;
          {datetime.now().strftime('%Y-%m-%d %H:%M ET')}
        </p>"""

        return self._send(
            subject=f"Daily Summary {date_str} | Portfolio ${portfolio_value:,.0f} | P&L ${daily_pnl:+,.0f}",
            html_body=html,
        )
