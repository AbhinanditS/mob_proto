import { useState } from 'react'
import Topbar from './components/Topbar'
import Canvas from './components/Canvas'
import Sidebar from './components/Sidebar'

export default function App() {
  const [posterSettings, setPosterSettings] = useState({
    layout: 0,
    fontSize: 64,
    fontWeight: 700,
    letterSpacing: -2,
    accentColor: '#B85C3A',
    darkPoster: false,
    horizontalMargin: 20,
    verticalGap: 16,
  })

  const [controlMode, setControlMode] = useState('default')
  const [inputValue, setInputValue] = useState('')
  const [isThinking, setIsThinking] = useState(false)
  const [editingField, setEditingField] = useState(null)

  const [posterText, setPosterText] = useState({
    headline: 'FORM.',
    subhead: 'conference.',
    date: '24 OCT',
    location: 'BENGALURU — 7PM IST',
    footer: 'camera: Fujifilm XS10  /  archive by form 2025',
  })

  return (
    <div className="flex h-screen w-screen flex-col" style={{ background: '#0E0E0E' }}>
      <Topbar />
      <div className="flex flex-1 min-h-0">
        <Canvas
          posterSettings={posterSettings}
          posterText={posterText}
          setPosterText={setPosterText}
          editingField={editingField}
          setEditingField={setEditingField}
        />
        <Sidebar
          controlMode={controlMode}
          setControlMode={setControlMode}
          posterSettings={posterSettings}
          setPosterSettings={setPosterSettings}
          inputValue={inputValue}
          setInputValue={setInputValue}
          isThinking={isThinking}
          setIsThinking={setIsThinking}
        />
      </div>
    </div>
  )
}
