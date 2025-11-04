<?php
header("Content-Type: application/json");
require_once("../config/conexion.php");

$sql = "SELECT * FROM dataset ORDER BY id DESC";
$result = $conn->query($sql);

$datos = [];

if ($result->num_rows > 0) {
    while ($fila = $result->fetch_assoc()) {
        $datos[] = $fila;
    }
    echo json_encode(["status" => "ok", "data" => $datos], JSON_UNESCAPED_UNICODE);
} else {
    echo json_encode(["status" => "error", "message" => "No hay datos registrados."]);
}

$conn->close();
?>
