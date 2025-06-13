import { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [recommendations, setRecommendations] = useState([]);

  useEffect(() => {
    // Получение рекомендаций
    fetch("http://localhost:5000/api/recommendations")
    .then(response => response.json())  // ← Вот это ключевой момент!
    .then(data => {
      if (Array.isArray(data)) {
        setRecommendations(data);
      } else {
        console.error("Ожидался массив, но пришло:", data);
        setRecommendations([]);
      }
    })
    .catch(err => {
      console.error("Ошибка запроса:", err);
      setRecommendations([]);
    });
  }, []);

  return (
  <div className="App">
    <header className="App-header">
      <h1>Система рекомендаций для улучшения качества услуг</h1>
    </header>

    <main className="App-content">
      <h2>Рекомендации по улучшению</h2>
      {recommendations.length === 0 ? (
        <p>Загрузка рекомендаций...</p>
      ) : (
        <ul>
          {recommendations.map((userRec, index) => (
            <li key={index}>
              <p><strong>Пользователь {userRec.user_id}:</strong></p>
              <ul>
                {userRec.recommendations.length > 0 ? (
                  userRec.recommendations.map((r, i) => <li key={i}>{r}</li>)
                ) : (
                  <li>Нет рекомендаций — всё хорошо</li>
                )}
              </ul>
              <hr />
            </li>
          ))}
        </ul>
      )}
    </main>


      <footer className="App-footer">
        <p>© 2025 Теплякова Юлия. Дипломная работа.</p>
      </footer>
    </div>
  );
}

export default App;
