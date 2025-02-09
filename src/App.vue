<template>
  <div id="app">
    <h1>Real Estate Game</h1>
    <component :is="currentView" />
  </div>
</template>

<script setup>
import SearchForm from './components/SearchForm.vue'
import ResultsList from './components/ResultsList.vue'

const currentView = ref('SearchForm')
</script>
