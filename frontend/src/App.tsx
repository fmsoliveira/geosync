import { useState } from "react";
import "./App.css";

function App() {
  const [response, setResponse] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [address, setAddress] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const res = await fetch("http://127.0.0.1:8080/crew", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ address }),
      });
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data = await res.json();
      setResponse(JSON.stringify(data, null, 2));
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError(String(err));
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Get Crew Data</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Enter address"
          value={address}
          onChange={(e) => setAddress(e.target.value)}
          style={{ marginRight: "1em" }}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Loading..." : "Fetch Crew"}
        </button>
      </form>
      {error && <p style={{ color: "red" }}>Error: {error}</p>}
      {response && (
        <pre
          style={{
            textAlign: "left",
            background: "#f4f4f4",
            padding: "1em",
            borderRadius: "8px",
            marginTop: "1em",
          }}
        >
          {response}
        </pre>
      )}
    </div>
  );
}

export default App;
