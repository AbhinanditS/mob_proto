import Poster from './Poster'

// The dark canvas zone that holds the floating poster, centered.
export default function Canvas(props) {
  return (
    <div
      className="relative flex items-center justify-center overflow-hidden"
      style={{ flex: '0 0 65%', background: '#0E0E0E' }}
    >
      <Poster {...props} />
    </div>
  )
}
