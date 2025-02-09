<template>
  <form @submit.prevent="handleSubmit">
    <input v-model="priceRange" placeholder="Price Range" />
    <select v-model="propertyType">
      <option>House</option>
      <option>Apartment</option>
      <option>Condo</option>
    </select>
    <button type="submit">Search</button>
  </form>
</template>

<script setup>
const priceRange = ref('')
const propertyType = ref('')

const handleSubmit = async () => {
  // Implement search logic
}
</script>
