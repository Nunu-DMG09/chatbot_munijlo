<?php
header("Content-Type: application/json");
require_once("../config/conexion.php");

// Leer mensaje del usuario
$input = json_decode(file_get_contents("php://input"), true);
$pregunta = trim($input["mensaje"] ?? "");

if (empty($pregunta)) {
    echo json_encode(["error" => "No se recibió ningún mensaje."]);
    exit;
}

// Paso 1: Buscar en la BD la información más relacionada
$sql = "SELECT respuesta FROM dataset WHERE pregunta LIKE ? OR respuesta LIKE ? LIMIT 3";
$stmt = $conn->prepare($sql);
$like = "%$pregunta%";
$stmt->bind_param("ss", $like, $like);
$stmt->execute();
$result = $stmt->get_result();

$contexto = "";
while ($row = $result->fetch_assoc()) {
    $contexto .= $row["respuesta"] . "\n";
}

if (empty($contexto)) {
    $contexto = "No hay información relacionada en la base de datos.";
}

$conn->close();

// Paso 2: Preparar el prompt para Ollama
$payload = [
    "model" => "llama3.2:1b",
    "messages" => [
        ["role" => "system", "content" => "Eres un asistente que responde en español y usa la información proporcionada."],
        ["role" => "user", "content" => "Contexto: $contexto \n\nPregunta: $pregunta"]
    ]
];

// Paso 3: Llamar a Ollama (localhost:11434)
$ch = curl_init("http://localhost:11434/api/chat");
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_HTTPHEADER, ["Content-Type: application/json"]);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));

$response = curl_exec($ch);
curl_close($ch);

// Paso 4: Mostrar la respuesta del modelo
echo $response;
?>
