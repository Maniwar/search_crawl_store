<template>
  <div v-for="listing in listings" :key="listing.id">
    <h3>{{ listing.title }}</h3>
    <p>${{ listing.price }}</p>
    <button @click="handleOffer(listing)">Make Offer</button>
  </div>
</template>

<script setup>
const listings = ref([])
const handleOffer = async (listing) => {
  // Implement offer logic
}
</script>
