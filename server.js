const express = require('express')
const app = express()
const port = 3000

app.use(express.static('src'))

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/src/index.html')
})

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`)
})
