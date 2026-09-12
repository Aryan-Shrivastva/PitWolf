import React, { useState } from 'react'
import openingVideo from './assets/f1-opening-background.mp4'
import './pitwolf.css'
import { SimulationReplayView } from './components/SimulationReplayView'
import { StrategyDashboard } from './components/StrategyDashboard'

class AppErrorBoundary extends React.Component {
  state = { error: null }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('PitWolf page render failed', error, info)
  }

  render() {
    if (!this.state.error) return this.props.children
    return <main style={{ minHeight: '100vh', display: 'grid', placeItems: 'center', padding: 32, background: '#090d0f', color: '#eaf7f1', fontFamily: 'Manrope, sans-serif' }}>
      <section style={{ maxWidth: 620, border: '1px solid rgba(225,255,245,.2)', padding: 28, background: 'rgba(11,20,21,.92)' }}>
        <p style={{ color: '#ff7043', font: "9px 'DM Mono'", letterSpacing: '.12em' }}>PITWOLF / RECOVERABLE PAGE ERROR</p>
        <h1 style={{ margin: '12px 0', font: "600 34px 'Space Grotesk'" }}>This page could not render.</h1>
        <p style={{ color: '#9cafa7', lineHeight: 1.6 }}>A temporary data or page-state error was caught before it could blank the application. Refresh the app to return to a clean state.</p>
        <button type="button" onClick={() => window.location.reload()} style={{ marginTop: 12, padding: '10px 14px', border: '1px solid rgba(255,112,67,.6)', background: 'transparent', color: '#ff9b78', font: "9px 'DM Mono'", letterSpacing: '.08em', cursor: 'pointer' }}>REFRESH PITWOLF</button>
      </section>
    </main>
  }
}

function App() {
  const [page, setPage] = useState('landing')
  const [simRequest, setSimRequest] = useState(null)

  if (page === 'sim') {
    return <AppErrorBoundary><SimulationReplayView
      initialRequest={simRequest}
      onRequestConsumed={() => setSimRequest(null)}
      onOpenDashboard={() => setPage('dashboard')}
      onHome={() => { setSimRequest(null); setPage('landing') }}
    /></AppErrorBoundary>
  }
  if (page === 'dashboard') {
    return <AppErrorBoundary><StrategyDashboard
      onHome={() => setPage('landing')}
      onOpenSimulation={(request) => { setSimRequest(request || null); setPage('sim') }}
    /></AppErrorBoundary>
  }

  return <main className="pitwolf-landing">
    <video className="pitwolf-video" autoPlay muted loop playsInline preload="metadata" aria-hidden="true">
      <source src={openingVideo} type="video/mp4" />
    </video>
    <div className="pitwolf-video-shade" aria-hidden="true" />
    <div className="pitwolf-grain" aria-hidden="true" />
    <header className="pitwolf-header">
      <div className="pitwolf-wordmark"><span>✦</span><strong>PITWOLF <em>- THE STRATEGIST</em></strong></div>
      <div className="pitwolf-status"><i /> RACE INTELLIGENCE / ONLINE</div>
    </header>
    <section className="pitwolf-hero">
      <p className="pitwolf-eyebrow"><span /> ENERGY &amp; OVERTAKE INTELLIGENCE</p>
      <h1>Make the<br /><em>Strategic call.</em></h1>
      <p className="pitwolf-copy">focused race-strategy workspace for understanding when to attack, when to save, and when the next opportunity is worth waiting for.</p>
      <button className="pitwolf-cta" onClick={() => setPage('sim')}>
        OPEN PITWOLF SIMULATION <span>↗</span>
      </button>
    </section>
  </main>
}

export default App
